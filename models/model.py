"""
model.py  —  CrabTransformer Vision Transformer.

Changed from previous version
-------------------------------
CrabTransformer.__init__ accepts `channel_mask_indices` (list[int] | None)
and passes it to PatchEmbedding.  Everything else is unchanged.

Typical usage
-------------
    # Build dataset first, then read metadata from it:
    train_loader, val_loader, test_loader = get_dataloaders(...)
    ds = train_loader.dataset

    model = CrabTransformer(
        in_channels          = ds.in_channels,
        channel_mask_indices = ds.channel_mask_indices,
        ...
    )
"""

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from . import components


class CrabTransformer(nn.Module):
    """
    Vision Transformer for spatial snow-crab recruitment prediction.

    Input  : [B, in_channels, 50, 50]  multi-channel spatial grids
    Output : [B, 1,           50, 50]  predicted recruit density grid

    Parameters
    ----------
    grid_size            : int   — spatial grid (default 50)
    patch_size           : int   — patch size (default 5, → 100 patches)
    in_channels          : int   — number of input channels; read from
                                   dataset.in_channels for a given channel config
    embed_dim            : int   — transformer embedding dimension
    num_heads            : int   — attention heads
    num_layers           : int   — transformer blocks
    d_ff                 : int   — feed-forward hidden dimension
    dropout              : float
    channel_mask_indices : list[int] | None
        One integer per channel, pointing into the 6-slot temporal mask vector.
        Computed by data_helper.get_channel_info(); read from
        dataset.channel_mask_indices.  If None, the 17-channel default is used.
    """

    def __init__(self, grid_size=50, patch_size=5, in_channels=17,
                 embed_dim=128, num_heads=8, num_layers=6, d_ff=512,
                 dropout=0.1, channel_mask_indices=None):
        super().__init__()

        self.grid_size      = grid_size
        self.patch_size     = patch_size
        self.in_channels    = in_channels
        self.embed_dim      = embed_dim
        self.num_heads      = num_heads
        self.num_layers     = num_layers
        self.d_ff           = d_ff
        self.dropout        = dropout
        self.n_patches      = (grid_size // patch_size) ** 2
        self.patch_grid_size = grid_size // patch_size

        # --- Sub-modules ---------------------------------------------------
        self.patch_embed = components.PatchEmbedding(
            grid_size=grid_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            dropout=dropout,
            channel_mask_indices=channel_mask_indices,
        )
        self.position_encode  = components.PositionalEncoding2D(self.n_patches, embed_dim)
        self.temporal_encode  = components.TemporalEncoding(embed_dim, max_years=30,
                                                            precompute_years=50)
        self.embedding_dropout = nn.Dropout(dropout)

        self.transformer_blocks = nn.ModuleList([
            components.TransformerBlock(embed_dim, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.norm    = nn.LayerNorm(embed_dim)
        self.decoder = components.SpatialDecoder(embed_dim, self.patch_grid_size)

        self._init_weights()

    # ------------------------------------------------------------------

    def forward(self, x, year_indices, temporal_mask=None,
                return_attention=False, spatial_mask=None):
        """
        Parameters
        ----------
        x             : [B, in_channels, 50, 50]
        year_indices  : [B]  integer year indices
        temporal_mask : [B, 6]  0/1 validity flags per temporal slot
        return_attention : bool
        spatial_mask  : [B, 50, 50] or [B, 1, 50, 50]  land mask

        Returns
        -------
        predictions [B, 1, 50, 50]  (+ attention_maps if return_attention)
        """
        B = x.shape[0]
        attention_maps = [] if return_attention else None

        x = self.patch_embed(x, mask=temporal_mask)     # [B, P, D]
        x = self.position_encode(x)
        x = self.temporal_encode(x, year_indices)
        x = self.embedding_dropout(x)

        for block in self.transformer_blocks:
            if return_attention:
                x, attn = block(x, return_attention=True)
                attention_maps.append(attn)
            else:
                x = block(x)

        x = self.norm(x)                                # [B, P, D]

        # Reshape to spatial grid for decoder
        x = x.transpose(1, 2)                           # [B, D, P]
        x = x.view(B, self.embed_dim,
                   self.patch_grid_size,
                   self.patch_grid_size)                # [B, D, 10, 10]
        x = self.decoder(x)                             # [B, 1, 50, 50]
        x = F.softplus(x)                               # enforce non-negativity

        if spatial_mask is not None:
            if spatial_mask.dim() == 3 and x.dim() == 4:
                spatial_mask = spatial_mask.unsqueeze(1)
            x = x * spatial_mask

        if return_attention:
            return x, attention_maps
        return x

    # ------------------------------------------------------------------

    def _init_weights(self):
        print("Initialising model weights …")
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

        if hasattr(self.decoder, 'conv_out'):
            nn.init.constant_(self.decoder.conv_out.bias, 0.0)
            print("  Decoder conv_out bias initialised to 0.0 "
                  "(will be overridden by warm-start in train.py)")
        print("  Weight initialisation complete.")

    # ------------------------------------------------------------------
    #  Attention visualisation helpers (unchanged from previous version)
    # ------------------------------------------------------------------

    '''    def visualize_attention(self, attention_maps, layer_idx=0, sample_idx=0,
                            patch_idx=None, save_path=None, top_k=10):
        attn = attention_maps[layer_idx][sample_idx]
        if patch_idx is None:
            patch_idx = self.n_patches // 2

        attn_avg     = attn.mean(dim=0)
        attn_weights = attn_avg[patch_idx].cpu().numpy()
        attn_grid    = attn_weights.reshape(self.patch_grid_size, self.patch_grid_size)
        top_indices  = np.argsort(attn_weights)[-top_k:]
        top_coords   = [(i // self.patch_grid_size, i % self.patch_grid_size)
                        for i in top_indices]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        ax = axes[0]
        im = ax.imshow(attn_grid, cmap='hot', interpolation='nearest')
        ax.set_title(f'Attention (Layer {layer_idx}, Patch {patch_idx})')
        src_r, src_c = patch_idx // self.patch_grid_size, patch_idx % self.patch_grid_size
        ax.plot(src_c, src_r, 'b*', markersize=20, markeredgecolor='white',
                markeredgewidth=2, label='Source')
        for r, c in top_coords[-5:]:
            ax.plot(c, r, 'co', markersize=10, markeredgecolor='white')
        ax.legend()
        plt.colorbar(im, ax=ax)

        ax = axes[1]
        for h in range(attn.shape[0]):
            vals = np.sort(attn[h, patch_idx].cpu().numpy())[-top_k:]
            ax.plot(range(top_k), vals, 'o-', label=f'H{h}', alpha=0.7)
        ax.set_title('Top-K by head')
        ax.legend(ncol=2, fontsize=8)

        ax = axes[2]
        top_mask = np.zeros((self.patch_grid_size, self.patch_grid_size))
        for r, c in top_coords:
            top_mask[r, c] = 1
        ax.imshow(attn_grid, cmap='gray', alpha=0.3)
        ax.imshow(top_mask,  cmap='Reds', alpha=0.6)
        ax.plot(src_c, src_r, 'b*', markersize=20, markeredgecolor='white',
                markeredgewidth=2)
        ax.set_title(f'Top-{top_k} attended patches')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig'''