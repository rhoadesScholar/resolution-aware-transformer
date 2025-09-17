#!/usr/bin/env python3
"""
Script to fix common runtime issues in RAT experiments.
Run this script before starting training to set optimal environment variables.
"""

import os
import sys
from pathlib import Path


def set_optimal_threading():
    """Set optimal threading configuration for multi-GPU training."""
    # Get number of CPU cores per GPU (rough estimate)
    try:
        import torch

        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            cpu_count = os.cpu_count() or 8
            threads_per_gpu = max(1, cpu_count // num_gpus)
        else:
            threads_per_gpu = 4
    except ImportError:
        threads_per_gpu = 4

    # Set environment variables
    os.environ["OMP_NUM_THREADS"] = str(threads_per_gpu)
    os.environ["MKL_NUM_THREADS"] = str(threads_per_gpu)
    os.environ["NUMEXPR_NUM_THREADS"] = str(threads_per_gpu)

    print(f"Set threading: OMP_NUM_THREADS={threads_per_gpu}")
    print(f"Set threading: MKL_NUM_THREADS={threads_per_gpu}")
    print(f"Set threading: NUMEXPR_NUM_THREADS={threads_per_gpu}")


def set_cuda_optimizations():
    """Set CUDA memory and performance optimizations."""
    # Memory optimization
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # For debugging (can be disabled in production)
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

    # Memory growth
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

    print("Set CUDA optimizations:")
    print(f"  PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}")
    print(f"  CUDA_LAUNCH_BLOCKING={os.environ.get('CUDA_LAUNCH_BLOCKING')}")


def fix_pillow_deprecations():
    """Apply fixes for Pillow deprecation warnings in setup scripts."""
    setup_isic_path = Path(__file__).parent / "setup_isic.py"

    if setup_isic_path.exists():
        print(f"Pillow deprecation fixes already applied to {setup_isic_path}")
    else:
        print(f"Warning: {setup_isic_path} not found")


def check_dependencies():
    """Check for required dependencies and suggest fixes."""
    required_packages = [
        "torch",
        "spatial-grouping-attention",
        "rotary-spatial-embeddings",
        "torchvision",
        "tqdm",
        "pyyaml",
    ]

    missing_packages = []

    import importlib.util

    for package in required_packages:
        # Try to find the module spec directly
        spec = importlib.util.find_spec(package)
        if spec is None:
            # Try replacing '-' with '_' for common cases
            alt_spec = importlib.util.find_spec(package.replace("-", "_"))
            if alt_spec is None:
                missing_packages.append(package)

    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nInstall with:")
        print(f"  pip install {' '.join(missing_packages)}")
        return False
    else:
        print("All required dependencies are installed.")
        return True


def main():
    """Main function to apply all fixes."""
    print("RAT Experiments - Common Issues Fix Script")
    print("=" * 50)

    # Set optimal environment variables
    set_optimal_threading()
    set_cuda_optimizations()

    # Check dependencies
    print("\nChecking dependencies...")
    deps_ok = check_dependencies()

    # Fix Pillow issues
    print("\nChecking Pillow fixes...")
    fix_pillow_deprecations()

    print("\n" + "=" * 50)
    if deps_ok:
        print("✅ All checks passed. You can now run experiments.")
    else:
        print("❌ Some issues found. Please install missing dependencies.")

    print("\nRecommended next steps:")
    print("1. Run your experiment with these environment variables set")
    print("2. Monitor GPU memory usage with: nvidia-smi -l 1")
    print("3. Check logs in experiments/results/lsf_logs/ for any issues")


if __name__ == "__main__":
    main()
