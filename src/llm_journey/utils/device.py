"""Device utilities for CPU/GPU selection."""

import torch


def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Get the best available device for training/inference.

    Args:
        prefer_cuda: If True, use CUDA if available

    Returns:
        torch.device: The selected device
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
