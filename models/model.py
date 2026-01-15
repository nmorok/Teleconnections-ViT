import numpy as np
import pandas as pd
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import components



class CrabTransformer(nn.Module):
    """
    Docstring for CrabTransformer
    """
    def __init__(self, grid_size=50, patch_size=5, in_channels=1, embed_dim=128, num_heads=8, num_layers=6, d_ff=512, mask=None, dropout=0.1):
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

        self.patch_embed = components.PatchEmbedding(self.grid_size, self.patch_size, self.in_channels, self.embed_dim)
        self.position_encode = components.PositionalEncoding2D(self.n_patches, self.embed_dim)

        self.transformer_block = components.TransformerBlock(self.embed_dim, self.num_heads, self.d_ff, self.dropout)

        #self.decoder

    def forward(self, x):
        """
        Docstring for forward
        
        :param self: Description
        :param x: Description
        """

        # Convert grid to patch embeddings
        x = self.patch_embed(x)  # [batch_size, num_patches, embed_dim]

        # Add positional encoding
        x = self.position_encode(x)  # [batch_size, num_patches, embed_dim]

        # Pass through transformer layers
        for _ in range(self.num_layers):
            x = self.transformer_block(x, self.mask)  # [batch_size, num_patches, embed_dim]

        #x = self.decoder(x)  # [batch_size, num_patches, embed_dim]

        return x  # [batch_size, num_patches, embed_dim]
    
    def get_attention_maps(self, x):
        """
        Docstring for get_attention_maps
        
        :param self: Description
        :param x: Description
        """

        attention_maps = []

        # Convert grid to patch embeddings
        x = self.patch_embed(x)  # [batch_size, num_patches, embed_dim]

        # Add positional encoding
        x = self.position_encode(x)  # [batch_size, num_patches, embed_dim]

        # Pass through transformer layers and collect attention maps
        for layer in self.transformer_block.layers:
            x, attn_map = layer.get_attention_map(x, self.mask)  # [batch_size, num_patches, embed_dim], [batch_size, heads, patches, patches]
            attention_maps.append(attn_map)

        return attention_maps  # List of attention maps from each layer
    
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
                    


        return 




        