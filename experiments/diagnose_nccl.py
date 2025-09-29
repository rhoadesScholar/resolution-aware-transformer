#!/usr/bin/env python3
"""
NCCL connectivity diagnostic script for RAT distributed training.
Run this before submitting large training jobs to check for network issues.
"""

import os
import sys
import socket
import time
import subprocess
from pathlib import Path


def check_cuda_availability():
    """Check CUDA and GPU availability."""
    try:
        import torch

        if not torch.cuda.is_available():
            print("❌ CUDA is not available")
            return False

        gpu_count = torch.cuda.device_count()
        print(f"✅ CUDA available with {gpu_count} GPUs")

        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"   GPU {i}: {gpu_name}")

        return True
    except ImportError:
        print("❌ PyTorch not available")
        return False


def check_nccl_environment():
    """Check NCCL environment variables."""
    print("\\n=== NCCL Environment Check ===")

    nccl_vars = [
        "NCCL_TIMEOUT_S",
        "NCCL_SOCKET_TIMEOUT",
        "NCCL_CONNECT_TIMEOUT",
        "NCCL_NET_RETRY_COUNT",
        "NCCL_SOCKET_IFNAME",
        "NCCL_DEBUG",
    ]

    for var in nccl_vars:
        value = os.environ.get(var, "Not set")
        print(f"   {var}: {value}")


def check_network_interfaces():
    """Check available network interfaces."""
    print("\\n=== Network Interfaces ===")
    try:
        result = subprocess.run(["ip", "addr", "show"], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.split("\\n")
            for line in lines:
                if "inet " in line and not "127.0.0.1" in line:
                    print(f"   {line.strip()}")
        else:
            print("   Could not get network interface information")
    except Exception as e:
        print(f"   Error checking network interfaces: {e}")


def test_distributed_communication():
    """Test basic distributed communication setup."""
    print("\\n=== Testing Distributed Communication ===")

    try:
        import torch
        import torch.distributed as dist

        # Set environment variables for testing
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "12355")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")

        # Try to initialize process group
        if not dist.is_initialized():
            print("   Initializing test process group...")
            dist.init_process_group(
                backend="nccl" if torch.cuda.is_available() else "gloo",
                init_method="env://",
                world_size=1,
                rank=0,
                timeout=torch.timedelta(seconds=30),
            )
            print("   ✅ Process group initialized successfully")

            # Test a simple all-reduce operation
            if torch.cuda.is_available():
                tensor = torch.ones(1).cuda()
                dist.all_reduce(tensor)
                print("   ✅ Basic NCCL all-reduce operation successful")

            dist.destroy_process_group()
            print("   ✅ Process group destroyed cleanly")
        else:
            print("   ✅ Distributed backend already initialized")

    except Exception as e:
        print(f"   ❌ Distributed communication test failed: {e}")
        return False

    return True


def check_scratch_directory():
    """Check scratch directory availability."""
    print("\\n=== Scratch Directory Check ===")

    user = os.environ.get("USER", "unknown")
    scratch_dir = f"/scratch/{user}"

    if os.path.exists(scratch_dir):
        if os.access(scratch_dir, os.W_OK):
            print(f"   ✅ Scratch directory writable: {scratch_dir}")

            # Test write performance
            test_file = Path(scratch_dir) / "nccl_test.tmp"
            try:
                start_time = time.time()
                test_file.write_text("test" * 1000)  # 4KB test
                write_time = time.time() - start_time
                test_file.unlink()
                print(f"   ✅ Write performance: {write_time:.3f}s for 4KB")
            except Exception as e:
                print(f"   ⚠️  Write test failed: {e}")
        else:
            print(f"   ❌ Scratch directory not writable: {scratch_dir}")
    else:
        print(f"   ❌ Scratch directory not found: {scratch_dir}")


def main():
    """Run all diagnostic checks."""
    print("=== RAT NCCL Connectivity Diagnostic ===")
    print(f"Hostname: {socket.gethostname()}")
    print(f"Python: {sys.version}")
    print(f"Working directory: {os.getcwd()}")

    # Run checks
    cuda_ok = check_cuda_availability()
    check_nccl_environment()
    check_network_interfaces()
    check_scratch_directory()

    if cuda_ok:
        comm_ok = test_distributed_communication()
    else:
        comm_ok = False

    # Summary
    print("\\n=== Diagnostic Summary ===")
    if cuda_ok and comm_ok:
        print("✅ All checks passed - distributed training should work")
        print("You can proceed with submitting your training job.")
        return 0
    else:
        print("❌ Some checks failed - distributed training may have issues")
        print("Consider using the simple runner or fixing the identified issues.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
