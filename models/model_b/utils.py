"""Utility functions for device management and seed setting."""
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Get available device with fallback to CPU.

    Returns:
        torch.device: CUDA if available, otherwise CPU.
    """
    if torch.cuda.is_available():
        try:
            # Test CUDA availability
            torch.cuda.get_device_name(0)
            # Test with Conv2d like the ViT model will use
            from torch import nn
            test_input = torch.randn(1, 3, 32, 32).cuda()
            test_conv = nn.Conv2d(3, 16, kernel_size=3).cuda()
            _ = test_conv(test_input)
            del test_input, test_conv
            torch.cuda.empty_cache()
            return torch.device("cuda")
        except RuntimeError:
            print("WARNING: CUDA detected but not usable. Using CPU.")
            return torch.device("cpu")
    return torch.device("cpu")
