"""Character-level dataset for language modeling."""

import torch
from torch.utils.data import Dataset


class CharDataset(Dataset):
    """
    Character-level dataset for language modeling.
    
    Builds a vocabulary from text and provides character-level tokenization.
    Returns shifted sequences (x, y) where y is x shifted by one position.
    """
    
    def __init__(self, text: str, block_size: int = 128):
        """
        Initialize the dataset from text.
        
        Args:
            text: Input text corpus
            block_size: Length of each training sequence
        """
        self.block_size = block_size
        
        # Build vocabulary from unique characters
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        
        # Encode the entire text
        self.data = torch.tensor([self.char_to_idx[ch] for ch in text], dtype=torch.long)
        
    def __len__(self):
        """Number of training examples (blocks)."""
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx: int):
        """
        Get a training pair (x, y) where y is x shifted by one.
        
        Args:
            idx: Index of the block
            
        Returns:
            tuple: (x, y) tensors of shape (block_size,)
        """
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y
    
    def encode(self, text: str) -> list[int]:
        """
        Encode text to token indices.
        
        Args:
            text: Text to encode
            
        Returns:
            List of token indices
        """
        return [self.char_to_idx[ch] for ch in text]
    
    def decode(self, indices: list[int]) -> str:
        """
        Decode token indices to text.
        
        Args:
            indices: List of token indices
            
        Returns:
            Decoded text string
        """
        return "".join([self.idx_to_char[i] for i in indices])
