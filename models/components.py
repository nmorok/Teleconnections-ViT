"""
components.py  —  building blocks for CrabTransformer.

Changed from previous version
-------------------------------
PatchEmbedding now accepts a `channel_mask_indices` argument (list[int],
length = in_channels).  Each entry is an index into the temporal-mask
vector (0-5), telling the layer which mask slot to apply to that channel.

This replaces the previous hardcoded if/elif block that assumed channels were
always laid out as [sp_current, sp_hist×5, rec_hist×5, temp_current, temp_hist×5].
Now any channel-group combination built by data_helper.get_channel_info() works
automatically — just pass channel_mask_indices from the dataset object through
CrabTransformer into PatchEmbedding.

Everything else (PositionalEncoding2D, TemporalEncoding, MultiHeadAttention,
FeedForward, TransformerBlock, SpatialDecoder) is unchanged.
"""

import numpy as np
import pandas as pd
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
#  PATCH EMBEDDING
# ============================================================

class PatchEmbedding(nn.Module):
    """
    Fully separate per-channel patch embeddings.

    Each input channel gets its own learnable linear projection
    (patch_size² → embed_dim_per_channel), followed by LayerNorm and Dropout.
    The per-channel embeddings are concatenated and projected to embed_dim.

    Temporal masking is applied per-channel using `channel_mask_indices`:
    a list of integers (0-5), one per channel, indicating which slot of the
    temporal mask vector (length 6: [current_year, t-1, …, t-5]) to use.
    A mask value of 0 zeros out the channel embedding completely.

    Parameters
    ----------
    grid_size           : int   — spatial grid size (default 50)
    patch_size          : int   — patch size (default 5)
    in_channels         : int   — total input channels (varies by channel config)
    embed_dim           : int   — final embedding dimension (default 128)
    dropout             : float — dropout probability (default 0.1)
    channel_mask_indices: list[int] | None
        Length must equal in_channels.  Each entry is 0-5, pointing into the
        temporal mask vector.  If None, a default mapping matching the full
        17-channel layout is used (backwards-compatible).
    """

    def __init__(self, grid_size=50, patch_size=5, in_channels=17,
                 embed_dim=128, dropout=0.1, channel_mask_indices=None):
        super().__init__()
        self.grid_size   = grid_size
        self.patch_size  = patch_size
        self.in_channels = in_channels
        self.num_patches = (grid_size // patch_size) ** 2

        # -- Validate / store channel_mask_indices -------------------------
        if channel_mask_indices is None:
            # Backwards-compatible default for the original 17-channel layout:
            # [sp_curr, sp_t-1…t-5, rec_t-1…t-5, temp_curr, temp_t-1…t-5]
            channel_mask_indices = [0, 1, 2, 3, 4, 5,   # spawners (6)
                                    1, 2, 3, 4, 5,       # recruits (5)
                                    0, 1, 2, 3, 4, 5]    # temp     (6)
        assert len(channel_mask_indices) == in_channels, (
            f"channel_mask_indices length ({len(channel_mask_indices)}) "
            f"must equal in_channels ({in_channels})"
        )
        # Store as a plain list (not a tensor) so it survives model.save/load
        self.channel_mask_indices = channel_mask_indices

        # -- Per-channel projections ---------------------------------------
        self.embed_dim_per_channel = math.ceil(embed_dim / in_channels)
        self.channel_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(patch_size * patch_size, self.embed_dim_per_channel),
                nn.LayerNorm(self.embed_dim_per_channel),
                nn.Dropout(dropout),
            )
            for _ in range(in_channels)
        ])

        # -- Final combination projection ----------------------------------
        self.combine = nn.Linear(in_channels * self.embed_dim_per_channel, embed_dim)

    def forward(self, x, mask=None):
        """
        Parameters
        ----------
        x    : [B, in_channels, 50, 50]
        mask : [B, 6]   temporal mask (0 = missing, 1 = valid)
                         index 0 = current year, 1-5 = lookbacks t-1…t-5

        Returns
        -------
        [B, num_patches, embed_dim]
        """
        batch_size = x.shape[0]

        # Reshape to patches: [B, in_channels, 50, 50] → [B, num_patches, in_channels, patch²]
        patches = x.unfold(2, self.patch_size, self.patch_size)
        patches = patches.unfold(3, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(batch_size, self.num_patches, self.in_channels, -1)

        # Per-channel projection + masking
        channel_embeds = []
        for c in range(self.in_channels):
            ch_patch  = patches[:, :, c, :]                       # [B, P, patch²]
            ch_embed  = self.channel_projections[c](ch_patch)     # [B, P, D_per_ch]

            if mask is not None:
                # Look up the correct temporal mask slot for this channel
                midx = self.channel_mask_indices[c]               # int in [0, 5]
                m    = mask[:, midx].view(batch_size, 1, 1)       # [B, 1, 1]
                ch_embed = ch_embed * m

            channel_embeds.append(ch_embed)

        # Concatenate then project to embed_dim
        combined   = torch.cat(channel_embeds, dim=-1)            # [B, P, in_ch × D_per_ch]
        embeddings = self.combine(combined)                        # [B, P, embed_dim]
        return embeddings


# ============================================================
#  POSITIONAL ENCODING (unchanged)
# ============================================================

class PositionalEncoding2D(nn.Module):
    """Learned 2-D positional embeddings, one per patch."""

    def __init__(self, n_patches=100, embedding_dim=128, scale=0.1):
        super().__init__()
        self.position_embeddings = nn.Parameter(
            torch.randn(1, n_patches, embedding_dim) * scale
        )

    def forward(self, x):
        return x + self.position_embeddings


# ============================================================
#  TEMPORAL ENCODING (unchanged)
# ============================================================

class TemporalEncoding(nn.Module):
    """
    Fixed sinusoidal temporal encoding added to all patch embeddings.

    PE(t, 2i)   = sin(t / 10000^(2i/d))
    PE(t, 2i+1) = cos(t / 10000^(2i/d))
    """

    def __init__(self, embed_dim=128, max_years=30, precompute_years=50):
        super().__init__()
        self.embed_dim        = embed_dim
        self.precompute_years = precompute_years

        pe       = torch.zeros(precompute_years, embed_dim)
        position = torch.arange(0, precompute_years, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe',       pe)
        self.register_buffer('div_term', div_term)

        print(f"TemporalEncoding: pre-computed 0–{precompute_years - 1}, "
              f"dynamic fallback beyond that.")

    def _compute_encoding(self, year_indices):
        batch_size = year_indices.shape[0]
        pe         = torch.zeros(batch_size, self.embed_dim, device=year_indices.device)
        position   = year_indices.float().unsqueeze(1)
        pe[:, 0::2] = torch.sin(position * self.div_term)
        pe[:, 1::2] = torch.cos(position * self.div_term)
        return pe

    def forward(self, x, year_indices):
        if torch.all(year_indices < self.precompute_years).item():
            temporal_emb = self.pe[year_indices]
        else:
            temporal_emb = self._compute_encoding(year_indices)
        return x + temporal_emb.unsqueeze(1)


# ============================================================
#  MULTI-HEAD SELF-ATTENTION (unchanged)
# ============================================================

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model=128, num_heads=8, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model  = d_model
        self.num_heads = num_heads
        self.d_head   = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        for w in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(w.weight)
            nn.init.zeros_(w.bias)

    def forward(self, x, return_attention=False):
        B, P, D = x.shape

        Q = self.W_q(x).view(B, P, self.num_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, P, self.num_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, P, self.num_heads, self.d_head).transpose(1, 2)

        scores  = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        out = torch.matmul(weights, V)
        out = out.transpose(1, 2).contiguous().view(B, P, D)
        out = self.W_o(out)

        if return_attention:
            return out, weights
        return out


# ============================================================
#  FEED-FORWARD NETWORK (unchanged)
# ============================================================

class FeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1   = nn.Linear(d_model, d_ff)
        self.linear2   = nn.Linear(d_ff, d_model)
        self.dropout   = nn.Dropout(dropout)
        self.activation = nn.GELU()

        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


# ============================================================
#  TRANSFORMER BLOCK (unchanged)
# ============================================================

class TransformerBlock(nn.Module):

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention   = MultiHeadAttention(d_model, num_heads, dropout)
        self.feedforward = FeedForward(d_model, d_ff, dropout)
        self.norm1       = nn.LayerNorm(d_model)
        self.norm2       = nn.LayerNorm(d_model)

    def forward(self, x, return_attention=False):
        normed = self.norm1(x)
        if return_attention:
            attn_out, attn_w = self.attention(normed, return_attention=True)
        else:
            attn_out = self.attention(normed)

        x = x + attn_out
        x = x + self.feedforward(self.norm2(x))

        if return_attention:
            return x, attn_w
        return x


# ============================================================
#  SPATIAL DECODER (unchanged)
# ============================================================

class SpatialDecoder(nn.Module):
    """
    Convolutional decoder: [B, embed_dim, 10, 10] → [B, 1, 50, 50].

    Three bilinear upsample + conv stages with reflection padding and
    GroupNorm (stable with small batch sizes).
    """

    def __init__(self, embed_dim=128, patch_grid_size=10):
        super().__init__()

        self.upsample1 = nn.Upsample(scale_factor=2,  mode='bilinear', align_corners=False)
        self.conv1     = nn.Conv2d(embed_dim, 64, 3, padding=1, padding_mode='reflect')
        self.bn1       = nn.GroupNorm(8, 64)
        self.act1      = nn.GELU()

        self.upsample2 = nn.Upsample(scale_factor=2,  mode='bilinear', align_corners=False)
        self.conv2     = nn.Conv2d(64, 32, 3, padding=1, padding_mode='reflect')
        self.bn2       = nn.GroupNorm(4, 32)
        self.act2      = nn.GELU()

        self.upsample3 = nn.Upsample(size=50, mode='bilinear', align_corners=False)
        self.conv3     = nn.Conv2d(32, 16, 3, padding=1, padding_mode='reflect')
        self.act3      = nn.GELU()

        self.conv_out  = nn.Conv2d(16, 1, 3, padding=1, padding_mode='reflect')

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(self.upsample1(x))))
        x = self.act2(self.bn2(self.conv2(self.upsample2(x))))
        x = self.act3(self.conv3(self.upsample3(x)))
        x = self.conv_out(x)
        return x