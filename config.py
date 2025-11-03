"""
Centralized configuration for device and dtype management.
This ensures consistent device handling across the entire codebase.
"""
import torch

# Check if CUDA is available and actually works
def get_device():
    """
    Get the appropriate device (CUDA if available and working, else CPU).
    Tests CUDA with a simple operation to ensure compatibility.
    """
    if torch.cuda.is_available():
        try:
            # Test if CUDA actually works by doing a simple operation
            test_tensor = torch.tensor([1.0], device="cuda")
            _ = torch.norm(test_tensor)
            return torch.device("cuda")
        except RuntimeError:
            # CUDA is available but doesn't work (e.g., incompatible GPU)
            print("Warning: CUDA is available but not compatible with this GPU. Using CPU instead.")
            return torch.device("cpu")
    else:
        return torch.device("cpu")

# Global device and dtype configuration
DEVICE = get_device()
DTYPE = torch.float32

# Aliases for compatibility (some files use lowercase)
device = DEVICE
dtype = DTYPE

