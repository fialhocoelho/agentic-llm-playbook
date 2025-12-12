"""Transformer block with pre-layer normalization."""

import torch
import torch.nn as nn
from llm_journey.models.mha import MultiHeadAttention


class TransformerBlock(nn.Module):
    """
    Pre-LN Transformer block.
    
    Architecture:
        x = x + MHA(LN(x))
        x = x + MLP(LN(x))
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Initialize transformer block.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            dropout: Dropout probability
        """
        super().__init__()
        
        # Multi-head attention
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # Feed-forward network (MLP)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through transformer block.
        
        Args:
            x: Input tensor of shape (B, T, d_model)
            attn_mask: Optional attention mask of shape (T, T)
            
        Returns:
            Output tensor of shape (B, T, d_model)
        """
        # Pre-LN attention with residual
        x = x + self.dropout1(self.attn(self.ln1(x), attn_mask))
        
        # Pre-LN MLP with residual
        x = x + self.dropout2(self.mlp(self.ln2(x)))
        
        return x
