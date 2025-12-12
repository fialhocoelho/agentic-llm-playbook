"""Scaled dot-product attention with causal masking."""

import torch
import torch.nn.functional as F


def causal_mask(T: int, device: torch.device) -> torch.Tensor:
    """
    Create a causal (lower-triangular) attention mask.
    
    Upper triangular values are set to large negative numbers so they 
    become ~0 after softmax.
    
    Args:
        T: Sequence length
        device: Device to create the mask on
        
    Returns:
        Mask tensor of shape (T, T) with -inf in upper triangle
    """
    # Create lower triangular mask (1s below diagonal, 0s above)
    mask = torch.tril(torch.ones(T, T, device=device))
    # Convert to attention mask: 0 -> -inf, 1 -> 0
    mask = mask.masked_fill(mask == 0, float('-inf'))
    mask = mask.masked_fill(mask == 1, 0.0)
    return mask


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute scaled dot-product attention.
    
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    
    Args:
        q: Query tensor of shape (..., T, d_k)
        k: Key tensor of shape (..., T, d_k)
        v: Value tensor of shape (..., T, d_v)
        attn_mask: Optional mask of shape (T, T) with 0s and -infs
        
    Returns:
        tuple: (output, attention_weights)
            - output: shape (..., T, d_v)
            - attention_weights: shape (..., T, T)
    """
    d_k = q.size(-1)
    
    # Compute attention scores: QK^T / sqrt(d_k)
    scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
    
    # Apply mask if provided
    if attn_mask is not None:
        scores = scores + attn_mask
    
    # Apply softmax to get attention weights
    attn_weights = F.softmax(scores, dim=-1)
    
    # Apply attention weights to values
    output = torch.matmul(attn_weights, v)
    
    return output, attn_weights
