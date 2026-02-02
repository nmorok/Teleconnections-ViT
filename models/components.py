import numpy as np
import pandas as pd
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Function for converting grid to patch

class PatchEmbedding(nn.Module):
    """ 
    Convert image to patch embeddings.

    Batch: number of replicates.
    Channel: how many different versions of the same grid (e.g., different features).


    Example: 50 x 50 grid with patch size of 5
    results in 100 patches of size 5 x 5, each flattened to a vector of size 25.

    """
    def __init__(self, grid_size=50, patch_size=5, in_channels=1, embed_dim=128):
        super().__init__()
        self.grid_size = grid_size
        self.patch_size = patch_size
        self.num_patches = (grid_size // patch_size) ** 2
        self.projection = nn.Linear(patch_size * patch_size * in_channels, embed_dim)

        # trying to fix weight initialization issues
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, x):
        # x: (batch_size, channels, height, width) = [B, 2, 50, 50]
        batch_size = x.shape[0]

        # Unfold into patches: [B, C, H, W] -> [B, C, n_patches_h, n_patches_w, patch_h, patch_w]
        patches = x.unfold(2, self.patch_size, self.patch_size)  # unfold height
        patches = patches.unfold(3, self.patch_size, self.patch_size)  # unfold width
        
        # patches shape: [B, 2, 10, 10, 5, 5]
        # Rearrange to: [B, n_patches_h, n_patches_w, C, patch_h, patch_w]
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        
        # Flatten patches: [B, 10, 10, 2, 5, 5] -> [B, 100, 50]
        patches = patches.view(batch_size, self.num_patches, -1)
        
        # Project patches to embedding dimension: [B, 100, 50] -> [B, 100, 128]
        embeddings = self.projection(patches)

        return embeddings



class PositionalEncoding2D(nn.Module):
    '''
    Add learned positional embeddings to patches. 
    Each of the 100 patches gets a unique position vector.
    '''
    def __init__(self, n_patches = 100, embedding_dim=128):
        super().__init__()
        #Learnable position embeddings
        self.position_embeddings = nn.Parameter(torch.randn(1, n_patches, embedding_dim) * 0.02) # initialize small random values

    def forward(self, x):
        # x: [batch, n_patches, embedding_dim]
        # add position embeddings (broadcast across batch)
        return x + self.position_embeddings 
    

class TemporalEncoding(nn.Module):
    """
    Sinusoidal temporal encoding for year indices.
    Fixed instead of learned for a first pass.

    PE(t, 2i)   = sin(t / 10000^(2i/d_model))
    PE(t, 2i+1) = cos(t / 10000^(2i/d_model))

    Allows for prediction
    """
    def __init__(self, embed_dim=128, max_years=30, precompute_years=50):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_years = max_years
        self.precompute_years = precompute_years
        
        # Pre-compute encodings for 0 to precompute_years-1
        pe = torch.zeros(precompute_years, embed_dim)
        position = torch.arange(0, precompute_years, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * 
            (-math.log(10000.0) / embed_dim)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
        self.register_buffer('div_term', div_term)
        
        print(f"Sinusoidal temporal encoding initialized:")
        print(f"  Pre-computed years: 0-{precompute_years-1}")
        print(f"  Can dynamically compute beyond year {precompute_years} ✓")
    
    def _compute_encoding(self, year_indices):
        """Compute encoding dynamically for arbitrary year indices."""
        batch_size = year_indices.shape[0]
        device = year_indices.device
        
        pe = torch.zeros(batch_size, self.embed_dim, device=device)
        position = year_indices.float().unsqueeze(1)
        
        pe[:, 0::2] = torch.sin(position * self.div_term)
        pe[:, 1::2] = torch.cos(position * self.div_term)
        
        return pe
    
    def forward(self, x, year_indices):
        """
        Args:
            x: Patch embeddings [batch, num_patches, embed_dim]
            year_indices: Year indices [batch] - any values
        
        Returns:
            x with temporal embeddings added [batch, num_patches, embed_dim]
        """
        # Check if all indices are in pre-computed range
        if torch.all(year_indices < self.precompute_years).item():
            # Fast path: use pre-computed encodings
            temporal_emb = self.pe[year_indices]
        else:
            # Slow path: compute dynamically
            temporal_emb = self._compute_encoding(year_indices)
        
        # Expand and add
        temporal_emb = temporal_emb.unsqueeze(1)
        
        return x + temporal_emb

    


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.
    
    This allows the model to jointly attend to information from different 
    representation subspaces at different positions.
    
    Args:
        d_model: Dimension of the model (e.g., 128)
        num_heads: Number of attention heads (e.g., 8)
        dropout: Dropout probability (default: 0.1)
    
    Example:
        Input:  [batch=2, patches=100, d_model=128]
        Output: [batch=2, patches=100, d_model=128]
    """
    def __init__(self, d_model=128, num_heads=8, dropout=0.1):
        super().__init__() # call the parent's init first

        # check that the model dimension is divisible by the number of heads so we can split up the data
        assert d_model % num_heads == 0, f'd_model ({d_model}) must be divisible by num_heads ({num_heads})'

        # save parameters so we can use them in forward()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads # dimension per head
        self.dropout_prob = dropout

        # creating the linear projections for queries, keys, and values
        self.W_q = nn.Linear(d_model, d_model) #Each `nn.Linear` is a fully connected layer that does: `output = input @ weight + bias`
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.W_o = nn.Linear(d_model, d_model) # after attention, we'll have [batch, patches, d_model] and we want the original dimension

        self.dropout = nn.Dropout(dropout) # nn.Dropout(p) randomly zeros out elements with probability p during training to prevent over fitting, to force model to not rely on any single connection. 

        # trying to fix weight initialization issues
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.W_o.weight)
        nn.init.zeros_(self.W_q.bias)
        nn.init.zeros_(self.W_k.bias)
        nn.init.zeros_(self.W_v.bias)
        nn.init.zeros_(self.W_o.bias)




    def forward(self, x, mask=None, return_attention=False):
        """
        Forward pass for multi-head attention.
    
        Args:
            x: Input tensor [batch_size, num_patches, d_model]
            mask: Optional mask [batch_size, num_patches] (for ignoring certain patches)
    
        Returns:
            Output tensor [batch_size, num_patches, d_model]
        """
        # get dimensions from the input
        batch_size, num_patches, d_model = x.shape

        # project inputs to queries, keys, values
        Q = self.W_q(x) # [batch_size, num_patches, d_model] -- linear() only operates on the last dimension.
        K = self.W_k(x) # [batch_size, num_patches, d_model]
        V = self.W_v(x) # [batch_size, num_patches, d_model]


        # reshape to split into multiple heads: [batch, patches, heads, d_head]
        Q = Q.view(batch_size, num_patches, self.num_heads, self.d_head) # view just changes the dimensions without changing any data. total elements must stay the same
        K = K.view(batch_size, num_patches, self.num_heads, self.d_head)
        V = V.view(batch_size, num_patches, self.num_heads, self.d_head)

        # Transpose to: [batch, heads, patches, d_head]
        Q = Q.transpose(1, 2)  # Swap patches and heads dimensions
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Compute attention scores: Q @ K^T
        # [batch, heads, patches, d_head] @ [batch, heads, d_head, patches]
        # = [batch, heads, patches, patches]
        scores = torch.matmul(Q, K.transpose(-2, -1)) # transposing only swaps the last two dimensions
        # scores[b, h, i, j] = "How much should patch i attend to patch j in head h?"

        # Scale by square root of d_head (for numerical stability)
        scores = scores / math.sqrt(self.d_head)


        # Apply mask if provided (set masked positions to large negative value)
        if mask is not None:
            # mask shape: [batch_size, num_patches]
            # Reshape to: [batch_size, 1, 1, num_patches] for broadcasting
            mask = mask.unsqueeze(1).unsqueeze(2) # unsqueeze adds dimension of shape 1. so that you can add different shaped tensors to each element. 
            scores = scores.masked_fill(mask == 0, -1e9) # so that the mask can be applied to all of the heads

        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)  # [batch, heads, patches, patches]
        
        # Apply dropout to attention weights
        attention_weights = self.dropout(attention_weights)

        # Apply attention weights to values
        # [batch, heads, patches, patches] @ [batch, heads, patches, d_head]
        # = [batch, heads, patches, d_head]
        output = torch.matmul(attention_weights, V)
        
        # Transpose back: [batch, heads, patches, d_head] → [batch, patches, heads, d_head]
        output = output.transpose(1, 2)

        # Combine heads: [batch, patches, heads, d_head] → [batch, patches, d_model]
        output = output.contiguous().view(batch_size, num_patches, self.d_model) # contiguous is for how the data is stored in memory and changes the order so you can use it.

        # Apply output projection -- this allows all of the heads to talk to each other. 
        output = self.W_o(output)  # [batch, patches, d_model] 

        if return_attention:
            return output, attention_weights
        return output
    

class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    
    Two linear transformations with a non-linearity in between:
    FFN(x) = Linear2(Activation(Linear1(x)))
    
    Args:
        d_model: Input/output dimension (e.g., 128)
        d_ff: Hidden dimension (typically 4 × d_model = 512)
        dropout: Dropout probability (default: 0.1)
    
    Example:
        Input:  [batch=2, patches=100, d_model=128]
        Hidden: [batch=2, patches=100, d_ff=512]
        Output: [batch=2, patches=100, d_model=128]
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()

        # First linear layer: expand dimension
        self.linear1 = nn.Linear(d_model, d_ff)
    
        # Second linear layer: contract back to original dimension
        self.linear2 = nn.Linear(d_ff, d_model)

        self.dropout = nn.Dropout(dropout)

        # Activation function (GELU is standard for transformers)
        self.activation = nn.GELU()

        # trying to fix weight initialization issues
        # ✓ FIXED: Initialize weights
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x):
        """
        Forward pass through feed-forward network.
    
        Args:
            x: Input tensor [batch_size, num_patches, d_model]
    
        Returns:
            Output tensor [batch_size, num_patches, d_model]
        """
        # expand, activate, contract, dropout
        x = self.linear1(x) # [B, P, 128] → [B, P, 512]
        x = self.activation(x) # Apply GELU
        x = self.dropout(x) # Regularization
        x = self.linear2(x) # [B, P, 512] → [B, P, 128]
        x = self.dropout(x) # more regularization

        return x


class TransformerBlock(nn.Module):
    """
    A single Transformer block with multi-head attention and feed-forward network.
    
    Structure (Pre-LN variant):
        x = x + Attention(LayerNorm(x))    # Skip connection
        x = x + FeedForward(LayerNorm(x))  # Skip connection
    
    Args:
        d_model: Model dimension (e.g., 128)
        num_heads: Number of attention heads (e.g., 8)
        d_ff: Feed-forward hidden dimension (e.g., 512)
        dropout: Dropout probability (default: 0.1)
    
    Example:
        Input:  [batch=2, patches=100, d_model=128]
        Output: [batch=2, patches=100, d_model=128]  # Same shape!
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        # Multi-head attention
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)

        # feed-forward network
        self.feedforward = FeedForward(d_model, d_ff, dropout)

        # layer normalization (one for attention, one for feed forward)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None, return_attention=False):
        """
        Forward pass through transformer block.
    
        Args:
            x: Input tensor [batch_size, num_patches, d_model]
            mask: Optional attention mask
            return_attention: Whether to return attention weights
    
        Returns:
            If return_attention=False: Output tensor
            If return_attention=True: (Output tensor, attention weights)
        """

        # attention block with skip connection (Pre-LN)
        # apply attention to normalized input, then add back original input
        normed = self.norm1(x)
        if return_attention:
            attn_output, attn_weights = self.attention(normed, mask, return_attention=True)
        else:
            attn_output = self.attention(normed, mask)
        
        x = x + attn_output

        # Feed-forward block with skip connection (Pre-LN)
        ff_output = self.feedforward(self.norm2(x))
        x = x + ff_output

        if return_attention:
            return x, attn_weights
        return x
    



class SpatialDecoder(nn.Module):
    """
    Decoder to reconstruct grid from patch embeddings.
    
    Args:
        grid_size: Size of input/output grid (default: 50)
        patch_size: Size of each patch (default: 5)
        in_channels: Number of input channels (default: 2)
        embed_dim: Embedding dimension (default: 128)
    """
    def __init__(self, embed_dim=128, patch_grid_size=10):
        super().__init__()

        # Stage 1: 10×10 → 20×20
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) # upsample takes increases the size
        self.conv1 = nn.Conv2d(embed_dim, 64, kernel_size=3, padding=1) # convolution reduces the number of channels. we go from 128 to 64. 
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.GELU()
        
        # Stage 2: 20×20 → 40×40
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.act2 = nn.GELU()
        
        # Stage 3: 40×40 → 50×50
        self.upsample3 = nn.Upsample(size=50, mode='bilinear', align_corners=False)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.act3 = nn.GELU()
        
        # Final output layer
        self.conv_out = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        

        # Initialize weights
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        # x: [B, embed_dim, 10, 10]
        
        x = self.upsample1(x)  # [B, 128, 20, 20]
        x = self.conv1(x)       # [B, 64, 20, 20]
        x = self.bn1(x)
        x = self.act1(x)
        
        x = self.upsample2(x)  # [B, 64, 40, 40]
        x = self.conv2(x)       # [B, 32, 40, 40]
        x = self.bn2(x)
        x = self.act2(x)
        
        x = self.upsample3(x)  # [B, 32, 50, 50]
        x = self.conv3(x)       # [B, 16, 50, 50]
        x = self.act3(x)
        
        x = self.conv_out(x)   # [B, 1, 50, 50]
        
        return x

        '''
        This essentially the same thing. except the below goes in the __init__ function as a single block.
        Doing the above to learn and see each step more clearly.
        self.decoder = nn.Sequential( # sequential is like a pipeline, running things in order
            # upsample from 10x10 to 25x25
            nn.ConvTranspose2d(embed_dim, 64, kernel_size=4, stride=2, padding=1), # [B, 128, 10, 10] -> [B, 64, 20, 20]  kernel size is like the filter. stride means to increase the size by that amount.
            nn.GELU(),
            nn.BatchNorm2d(64), # normalize across batch for stability -- for images normally.

            # upsample from 25x25 to 50x50
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(32),

            # Stage 3: 40×40 → 50×50 (use Upsample for exact size)
            nn.Upsample(size=50, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.GELU(),

            nn.Conv2d(16, 1, kernel_size=3, padding=1) # final output layer [B, 32, 50, 50] -> [B, 1, 50, 50]
        )'''
  

