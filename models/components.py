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

    def forward(self, x):
        # x: (batch_size, channels, height, width) = [B, 1, 50, 50]

        batch__size = x.shape[0]

        # Unfold into patches: [B, 1, 50, 50] -> [B, 100, 25]
        patches = x.unfold(2, self.patch_size, self.patch_size) # unfold height
        
        patches = patches.unfold(3, self.patch_size, self.patch_size) # unfold width
        
        patches = patches.contiguous().view(batch__size, -1, self.patch_size * self.patch_size) # flatten patches
        
        # Project patches to embedding dimension: [B, 100, 25] -> [B, 100, 128]
        embeddings = self.projection(patches)
        return embeddings



class PositionalEncoding2D(nn.Module):
    '''
    Add leardned positional embeddings to patches. 
    Each of the 100 patches gets a unique position vector.
    '''
    def __init__(self, n_patches = 100, embedding_dim=128):
        super().__init__()
        #Learnable position embeddings
        self.position_embeddings = nn.Parameter(torch.randn(1, n_patches, embedding_dim))

    def forward(self, x):
        # x: [batch, n_patches, embedding_dim]
        # add position embeddings (broadcast across batch)
        return x + self.position_embeddings 
    

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

    def forward(self, x, mask=None):
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

        # Apply output projection
        output = self.W_o(output)  # [batch, patches, d_model]

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

    def forward(self, x, mask=None):
        """
        Forward pass through transformer block.
    
        Args:
            x: Input tensor [batch_size, num_patches, d_model]
            mask: Optional attention mask
    
        Returns:
            Output tensor [batch_size, num_patches, d_model]
        """

        # attention block with skip connection (Pre-LN)
        # apply attention to normalized input, then add back original input
        attn_output = self.attention(self.norm1(x), mask)
        x = x + attn_output

        # Feed-forward block with skip connection (Pre-LN)
        # Apply feedforward to normalized input, then add back previous output
        ff_output = self.feedforward(self.norm2(x))
        x = x + ff_output

        return x