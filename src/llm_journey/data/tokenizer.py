"""Simple character-level tokenizer."""

from typing import List


class SimpleTokenizer:
    """
    A simple character-level tokenizer for educational purposes.

    This tokenizer maps characters to integers and vice versa.
    """

    def __init__(self, corpus: str):
        """
        Initialize the tokenizer with a text corpus.

        Args:
            corpus: Text corpus to build vocabulary from
        """
        chars = sorted(list(set(corpus)))
        self.vocab_size = len(chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text: str) -> List[int]:
        """
        Encode text to a list of token IDs.

        Args:
            text: Input text

        Returns:
            List of token IDs
        """
        return [self.char_to_idx[ch] for ch in text]

    def decode(self, tokens: List[int]) -> str:
        """
        Decode token IDs back to text.

        Args:
            tokens: List of token IDs

        Returns:
            Decoded text string
        """
        return "".join([self.idx_to_char[idx] for idx in tokens])
