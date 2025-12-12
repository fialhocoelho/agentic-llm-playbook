"""Utilities for reproducibility and random seeding."""

import random
import numpy as np
import torch


def seed_everything(seed: int = 42):
    """
    Set random seeds for Python, NumPy, and PyTorch for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Make PyTorch deterministic (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
