import numpy as np
import pandas as pd
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
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
    def __init__(self, grid_size=50, patch_size=5, in_channels=11, embed_dim=128, num_heads=8, num_layers=6, d_ff=512, dropout=0.1):
        super().__init__()

        # initialize all of the variables
        self.grid_size = grid_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.n_patches = (grid_size // patch_size) ** 2
        self.patch_grid_size = grid_size // patch_size  # e.g., 50/5=10

        self.patch_embed = components.PatchEmbedding(self.grid_size, self.patch_size, self.in_channels, self.embed_dim)
        self.position_encode = components.PositionalEncoding2D(self.n_patches, self.embed_dim)
        self.temporal_encode = components.TemporalEncoding(self.embed_dim, max_years=30, precompute_years=50)

        self.embedding_dropout = nn.Dropout(self.dropout)

        self.transformer_blocks = nn.ModuleList([
            components.TransformerBlock(self.embed_dim, self.num_heads, self.d_ff, self.dropout)
            for _ in range(self.num_layers)
        ])

        self.norm = nn.LayerNorm(self.embed_dim)

        self.decoder = components.SpatialDecoder(self.embed_dim, self.patch_grid_size)

        self._init_weights()


    def forward(self, x, year_indices, return_attention=False):
        """
        Args:
            x: Input spawner grids [batch, 1, 50, 50]
            year_indices: Year indices [batch]
            temporal_mask: Temporal mask [batch, num_patches] (default: None)
            return_attention: Whether to return attention weights
        
        Returns:
            predictions: Predicted recruitment grids [batch, 1, 50, 50]
            (optional) attention_weights: List of attention maps from each transformer block
        """
        # getting the batch size
        batch_size = x.shape[0]

        if return_attention:
            attention_maps = []

        # Convert grid to patch embeddings
        x = self.patch_embed(x)  # [batch_size, num_patches, embed_dim] [B, 1, 50, 50] → [B, 100, 128]

        # Add positional encoding
        x = self.position_encode(x)  # [batch_size, num_patches, embed_dim] [B, 100, 128]

        # Add temporal encoding
        x = self.temporal_encode(x, year_indices)  # [batch_size, num_patches, embed_dim] [B, 100, 128]

        # Apply dropout
        x = self.embedding_dropout(x)  # [batch_size, num_patches, embed

        # Pass through transformer layers
        for block in self.transformer_blocks:
            
            if return_attention:
                x, attn = block(x, return_attention=True)  # [batch_size, num_patches, embed_dim], [batch_size, heads, patches, patches]
                attention_maps.append(attn)
            else:
                x = block(x)  # [batch_size, num_patches, embed_dim]
        
        x = self.norm(x)  # [batch_size, num_patches, embed_dim]

        # Decode to output patches
        # need to reshape first. from [batch_size, num_patches, embed_dim] to [batch_size, embed_dim, num_patches per grid size (10), num_patches per grid size (10)]
        x = x.transpose(1, 2)
        x = x.view(batch_size, self.embed_dim, self.patch_grid_size, self.patch_grid_size) # input [batch, dimensions, n_patches, n_patches] like each pixel has 128 dimensions 
        x = self.decoder(x)  # [batch, channels, n_patches, n_patches] 

        #x = torch.sigmoid(x)  # [batch_size, 1, grid_size, grid_size]
        x = torch.softplus(x)  # [batch_size, 1, grid_size, grid_size]


        if return_attention:
            return x, attention_maps  # [batch_size, 1, grid_size, grid_size], List of attention maps
        else:
            return x  # [batch_size, num_patches, embed_dim]
    
    
    def visualize_attention(self, attention_maps, layer_idx=0, sample_idx=0, 
                        patch_idx=None, save_path='data/dummy/output', top_k=10):
        """
        Visualize attention patterns from transformer layers.
        
        Shows which patches the model attends to when processing a specific patch.
        
        Args:
            attention_maps: List of attention tensors from forward pass
                        Each: [batch, heads, patches, patches]
            layer_idx: Which transformer layer to visualize (0 to num_layers-1)
            sample_idx: Which sample in batch to visualize
            patch_idx: Which patch to show attention for (None = center patch)
            save_path: Where to save figure (None = display only)
            top_k: Number of top attended patches to highlight
        
        Returns:
            matplotlib figure
        """
        import matplotlib.pyplot as plt

        
        # Get attention map for specified layer and sample
        attn = attention_maps[layer_idx][sample_idx]  # [heads, patches, patches]
        
        # Default to center patch if not specified
        if patch_idx is None:
            patch_idx = self.n_patches // 2  # Center patch (patch 50 for 10×10)
        
        # Average over all attention heads
        attn_avg = attn.mean(dim=0)  # [patches, patches]
        
        # Get attention weights for the specified patch
        attn_weights = attn_avg[patch_idx]  # [patches] - how much patch_idx attends to others
        
        # Convert to numpy
        attn_weights = attn_weights.cpu().numpy()
        
        # Reshape to 2D grid
        attn_grid = attn_weights.reshape(self.patch_grid_size, self.patch_grid_size)
        
        # Get top-k attended patches
        top_indices = np.argsort(attn_weights)[-top_k:]
        top_coords = [(idx // self.patch_grid_size, idx % self.patch_grid_size) 
                    for idx in top_indices]
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # --- Plot 1: Attention heatmap ---
        ax = axes[0]
        im = ax.imshow(attn_grid, cmap='hot', interpolation='nearest')
        ax.set_title(f'Attention Map (Layer {layer_idx}, Patch {patch_idx})', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Patch Column')
        ax.set_ylabel('Patch Row')
        
        # Mark the source patch
        source_row = patch_idx // self.patch_grid_size
        source_col = patch_idx % self.patch_grid_size
        ax.plot(source_col, source_row, 'b*', markersize=20, 
                markeredgecolor='white', markeredgewidth=2, label='Source Patch')
        
        # Mark top attended patches
        for i, (row, col) in enumerate(top_coords[-5:]):  # Top 5
            ax.plot(col, row, 'co', markersize=10, markeredgecolor='white', 
                    markeredgewidth=1.5)
        
        ax.legend(loc='upper right')
        plt.colorbar(im, ax=ax, label='Attention Weight')
        ax.grid(True, alpha=0.3)
        
        # --- Plot 2: Attention per head ---
        ax = axes[1]
        n_heads = attn.shape[0]
        head_attentions = attn[:, patch_idx, :].cpu().numpy()  # [heads, patches]
        
        # Show top-k patches for each head
        for head_idx in range(n_heads):
            head_attn = head_attentions[head_idx]
            top_vals = np.sort(head_attn)[-top_k:]
            ax.plot(range(top_k), top_vals, 'o-', label=f'Head {head_idx}', alpha=0.7)
        
        ax.set_title('Top-K Attention by Head', fontsize=14, fontweight='bold')
        ax.set_xlabel('Rank (highest to lowest)')
        ax.set_ylabel('Attention Weight')
        ax.legend(ncol=2, fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # --- Plot 3: Spatial distribution ---
        ax = axes[2]
        
        # Create binary mask of top attended regions
        top_mask = np.zeros((self.patch_grid_size, self.patch_grid_size))
        for row, col in top_coords:
            top_mask[row, col] = 1
        
        # Overlay on attention grid
        ax.imshow(attn_grid, cmap='gray', alpha=0.3)
        ax.imshow(top_mask, cmap='Reds', alpha=0.6)
        
        # Draw grid lines
        for i in range(self.patch_grid_size + 1):
            ax.axhline(i - 0.5, color='black', linewidth=0.5, alpha=0.3)
            ax.axvline(i - 0.5, color='black', linewidth=0.5, alpha=0.3)
        
        # Mark source and top patches
        ax.plot(source_col, source_row, 'b*', markersize=20, 
                markeredgecolor='white', markeredgewidth=2)
        for row, col in top_coords:
            ax.plot(col, row, 'ro', markersize=8, markeredgecolor='white', 
                    markeredgewidth=1.5)
        
        ax.set_title(f'Top-{top_k} Attended Patches', fontsize=14, fontweight='bold')
        ax.set_xlabel('Patch Column')
        ax.set_ylabel('Patch Row')
        ax.set_xlim(-0.5, self.patch_grid_size - 0.5)
        ax.set_ylim(self.patch_grid_size - 0.5, -0.5)  # Flip y-axis
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Attention visualization saved: {save_path}")
        
        return fig


    def visualize_all_layers(self, attention_maps, sample_idx=0, patch_idx=None, 
                            save_dir='attention_maps'):
        """
        Visualize attention for all transformer layers at once.
        
        Args:
            attention_maps: List of attention tensors from forward pass
            sample_idx: Which sample in batch
            patch_idx: Which patch to show (None = center)
            save_dir: Directory to save figures
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        if patch_idx is None:
            patch_idx = self.n_patches // 2
        
        n_layers = len(attention_maps)
        
        fig, axes = plt.subplots(2, (n_layers + 1) // 2, figsize=(4 * n_layers, 8))
        axes = axes.flatten()
        
        for layer_idx in range(n_layers):
            ax = axes[layer_idx]
            
            # Get attention for this layer
            attn = attention_maps[layer_idx][sample_idx]  # [heads, patches, patches]
            attn_avg = attn.mean(dim=0)  # Average over heads
            attn_weights = attn_avg[patch_idx].cpu().numpy()
            attn_grid = attn_weights.reshape(self.patch_grid_size, self.patch_grid_size)
            
            # Plot
            im = ax.imshow(attn_grid, cmap='hot', interpolation='nearest')
            ax.set_title(f'Layer {layer_idx}', fontsize=12, fontweight='bold')
            
            # Mark source patch
            source_row = patch_idx // self.patch_grid_size
            source_col = patch_idx % self.patch_grid_size
            ax.plot(source_col, source_row, 'b*', markersize=15, 
                    markeredgecolor='white', markeredgewidth=2)
            
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im, ax=ax)
        
        # Hide unused subplots
        for idx in range(n_layers, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'Attention Maps Across All Layers (Patch {patch_idx})', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, f'all_layers_patch_{patch_idx}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ All layers visualization saved: {save_path}")
        
        return fig

    def _init_weights(self):
        """
        Initialize weights using Xavier/Glorot initialization.
        
        This is CRITICAL for preventing gradient explosion in deep networks.
        Without this, default PyTorch initialization can cause massive gradients.
        """
        print("Initializing model weights...")
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier uniform initialization for linear layers
                # This scales weights based on input/output dimensions
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
            elif isinstance(module, nn.Conv2d):
                # Kaiming initialization for convolutional layers
                # This is better for ReLU activations
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
            elif isinstance(module, nn.ConvTranspose2d):
                # Xavier for transposed convolutions (used in decoder)
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
                # Standard initialization for normalization layers
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
                
            elif isinstance(module, nn.Embedding):
                # Normal initialization for embeddings
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
        if hasattr(self.decoder, 'conv_out'):
            print("✓ Applying Bias Initialization Surgery to Decoder (-3.0)")
            nn.init.constant_(self.decoder.conv_out.bias, -3.0)
        else:
            print("⚠️ WARNING: Could not find 'conv_out' in decoder to apply bias fix.")
        
        print("✓ Weight initialization complete")




        