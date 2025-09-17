#!/bin/bash
# Environment setup script for RAT experiments.
# Source this script before running experiments to set optimal configurations.
#
# Usage:
#     source experiments/scripts/setup_environment.sh
#     # or
#     . experiments/scripts/setup_environment.sh
#

# Get number of GPUs and CPUs for optimal threading
if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi -L | wc -l)
else
    NUM_GPUS=1
fi

NUM_CPUS=$(nproc)
THREADS_PER_GPU=$((NUM_CPUS / NUM_GPUS))

# Ensure at least 1 thread per GPU, max 8
if [ $THREADS_PER_GPU -lt 1 ]; then
    THREADS_PER_GPU=1
elif [ $THREADS_PER_GPU -gt 8 ]; then
    THREADS_PER_GPU=8
fi

# Set optimal threading
export OMP_NUM_THREADS=$THREADS_PER_GPU
export MKL_NUM_THREADS=$THREADS_PER_GPU
export NUMEXPR_NUM_THREADS=$THREADS_PER_GPU

# CUDA optimizations
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_LAUNCH_BLOCKING="1"  # For debugging - disable in production

# Memory optimizations
export TF_FORCE_GPU_ALLOW_GROWTH="true"

# Distributed training optimizations
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1

echo "RAT Environment Setup Complete"
echo "=============================="
echo "Threading configuration:"
echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "  MKL_NUM_THREADS=$MKL_NUM_THREADS"
echo "  NUMEXPR_NUM_THREADS=$NUMEXPR_NUM_THREADS"
echo ""
echo "CUDA configuration:"
echo "  PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
echo "  CUDA_LAUNCH_BLOCKING=$CUDA_LAUNCH_BLOCKING"
echo ""
echo "Detected: $NUM_GPUS GPUs, $NUM_CPUS CPU cores"
echo "Using $THREADS_PER_GPU threads per GPU"
echo ""
echo "You can now run experiments with optimal settings."