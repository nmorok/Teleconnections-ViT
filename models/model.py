import numpy as np
import pandas as pd
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import components



class CrabTransformer(nn.Module):
    """
    Vision Transformer for predicting crab recruitment from spawner spatial data.
    
    Takes a 50x50 grid of spawner data and predicts a 50x50 grid of recruitment data.
    Uses patch-based attention to capture spatial relationships.
    
    Args:
        grid_size: Size of input/output grid (default: 50)
        patch_size: Size of each patch (default: 5)
        in_channels: Number of input channels (default: 2)
        embed_dim: Embedding dimension (default: 128)
        num_heads: Number of attention heads (default: 8)
        num_layers: Number of transformer blocks (default: 6)
        d_ff: Feed-forward hidden dimension (default: 512)
        dropout: Dropout probability (default: 0.1)
        output_type: Type of output - 'grid' or 'scalar' (default: 'grid')


        """
    def __init__(self, grid_size=50, patch_size=5, in_channels=2, embed_dim=128, num_heads=8, num_layers=6, d_ff=512, mask=None, dropout=0.1):
        super().__init__()

        # initialize all of the variables
        self.grid_size = grid_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.mask = mask
        self.dropout = dropout
        self.n_patches = (grid_size // patch_size) ** 2
        self.patch_grid_size = grid_size // patch_size  # e.g., 50/5=10

        self.patch_embed = components.PatchEmbedding(self.grid_size, self.patch_size, self.in_channels, self.embed_dim)
        self.position_encode = components.PositionalEncoding2D(self.n_patches, self.embed_dim)

        self.transformer_blocks = nn.ModuleList([
            components.TransformerBlock(self.embed_dim, self.num_heads, self.d_ff, self.dropout)
            for _ in range(self.num_layers)
        ])
        self.decoder = components.SpatialDecoder(self.embed_dim, self.patch_grid_size)


    def forward(self, x, return_attention=False):
        """
        Docstring for forward
        
        :param self: Description
        :param x: Description
        """
        # getting the batch size
        batch_size = x.shape[0]

        if return_attention:
            attention_maps = []

        # Convert grid to patch embeddings
        x = self.patch_embed(x)  # [batch_size, num_patches, embed_dim]

        # Add positional encoding
        x = self.position_encode(x)  # [batch_size, num_patches, embed_dim]

        # Pass through transformer layers
        for block in self.transformer_blocks:
            
            if return_attention:
                x, attn = block(x, self.mask, return_attention=True)  # [batch_size, num_patches, embed_dim], [batch_size, heads, patches, patches]
                attention_maps.append(attn)
            else:
                x = block(x, self.mask)  # [batch_size, num_patches, embed_dim]

        # Decode to output patches
        # need to reshape first. from [batch_size, num_patches, embed_dim] to [batch_size, embed_dim, num_patches, num_patches]
        x = x.transpose(1, 2)
        x = x.view(batch_size, self.embed_dim, self.patch_grid_size, self.patch_grid_size)
        x = self.decoder(x)  # [batch, channels, n_patches, n_patches]


        if return_attention:
            return x, attention_maps  # [batch_size, 1, grid_size, grid_size], List of attention maps
        else:
            return x  # [batch_size, num_patches, embed_dim]
    
    
    def visualize_attention(self, attention_maps, patch_indices):
        """
        Docstring for visualize_attention
        
        :param self: Description
        :param attention_maps: Description
        :param patch_indices: Description
        """

        # Visualize attention maps for specified patch indices
        for layer_idx, attn_map in enumerate(attention_maps):
            for head_idx in range(self.num_heads):
                for patch_idx in patch_indices:
                    attn_weights = attn_map[:, head_idx, patch_idx, :]  # [batch_size, patches]
                    # Reshape and visualize as needed
                    


        pass 





        