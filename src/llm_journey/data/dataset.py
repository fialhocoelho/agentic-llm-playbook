"""Dataset class for text data."""

import torch
from torch.utils.data import Dataset
from typing import List


class TextDataset(Dataset):
    """
    Simple text dataset for language modeling.

    This dataset provides sequences of tokens with a sliding window.
    """

    def __init__(self, tokens: List[int], seq_length: int):
        """
        Initialize the dataset.

        Args:
            tokens: List of token IDs
            seq_length: Length of sequences to generate
        """
        self.tokens = tokens
        self.seq_length = seq_length

    def __len__(self) -> int:
        """Return the number of sequences in the dataset."""
        return len(self.tokens) - self.seq_length

    def __getitem__(self, idx: int) -> tuple:
        """
        Get a sequence and its target.

        Args:
            idx: Index of the sequence

        Returns:
            Tuple of (input_sequence, target_sequence)
        """
        x = torch.tensor(self.tokens[idx : idx + self.seq_length], dtype=torch.long)
        y = torch.tensor(
            self.tokens[idx + 1 : idx + self.seq_length + 1], dtype=torch.long
        )
        return x, y
