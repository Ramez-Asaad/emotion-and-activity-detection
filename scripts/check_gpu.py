"""
GPU Check Script
===============

Verify CUDA/GPU availability.
"""

import torch

print("=" * 70)
print("GPU Configuration Check")
print("=" * 70)

print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    
    # Memory info
    print(f"\nGPU Memory:")
    print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"  Cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
else:
    print("\n⚠ CUDA is NOT available - PyTorch will use CPU")
    print("\nTo install PyTorch with CUDA support:")
    print("  pip uninstall torch torchvision torchaudio")
    print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

print("\n" + "=" * 70)
