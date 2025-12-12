"""GPT-style mini language model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from llm_journey.models.attention import causal_mask
from llm_journey.models.transformer import TransformerBlock


class GPTMini(nn.Module):
    """
    A minimal GPT-style language model.

    Architecture:
        - Token embeddings + learned positional embeddings
        - N transformer blocks
        - Layer norm + language modeling head
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        d_ff: int = 512,
        max_seq_len: int = 256,
        dropout: float = 0.1,
    ):
        """
        Initialize GPTMini.

        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            n_layers: Number of transformer blocks
            n_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Learned positional embeddings
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        self.dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

        # Final layer norm
        self.ln_f = nn.LayerNorm(d_model)

        # Language modeling head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying (optional but common)
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass.

        Args:
            idx: Input token indices of shape (B, T)
            targets: Optional target indices of shape (B, T) for loss computation

        Returns:
            tuple: (logits, loss)
                - logits: shape (B, T, vocab_size)
                - loss: scalar loss if targets provided, else None
        """
        B, T = idx.shape
        device = idx.device

        # Token and position embeddings
        tok_emb = self.token_embedding(idx)  # (B, T, d_model)
        pos = torch.arange(0, T, dtype=torch.long, device=device)  # (T,)
        pos_emb = self.position_embedding(pos)  # (T, d_model)

        x = self.dropout(tok_emb + pos_emb)  # (B, T, d_model)

        # Create causal mask
        mask = causal_mask(T, device)  # (T, T)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask)

        # Final layer norm
        x = self.ln_f(x)  # (B, T, d_model)

        # Language modeling head
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """
        Generate new tokens autoregressively.

        Args:
            idx: Starting token indices of shape (B, T)
            max_new_tokens: Number of new tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top-k most likely tokens

        Returns:
            Generated token indices of shape (B, T + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Crop to max sequence length
            idx_cond = (
                idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len :]
            )

            # Get predictions
            logits, _ = self(idx_cond)

            # Focus on last time step
            logits = logits[:, -1, :] / temperature  # (B, vocab_size)

            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Append to sequence
            idx = torch.cat([idx, idx_next], dim=1)  # (B, T+1)

        return idx
