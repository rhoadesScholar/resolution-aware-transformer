#!/bin/bash
#BSUB -J RAT_medseg
#BSUB -o results/lsf_logs/RAT_medseg_%J.out
#BSUB -e results/lsf_logs/RAT_medseg_%J.err
#BSUB -n 96
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4000,ngpus_excl_p=8]"
#BSUB -q gpu_h100
#BSUB -gpu "num=8"

# LSF job submission script for Resolution Aware Transformer training
# This script properly configures the distributed environment for Ray training

echo "=== LSF Job Information ==="
echo "Job ID: $LSB_JOBID"
echo "Job Name: $LSB_JOBNAME"
echo "Queue: $LSB_QUEUE"
echo "Hosts: $LSB_HOSTS"
echo "Number of processors: $LSB_DJOB_NUMPROC"
echo "Working directory: $(pwd)"
echo "Started at: $(date)"
echo ""

# Setup environment
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# NCCL network configuration for cluster stability
export NCCL_TIMEOUT_S=7200
export NCCL_SOCKET_TIMEOUT=7200
export NCCL_CONNECT_TIMEOUT=300
export NCCL_NET_RETRY_COUNT=5
export NCCL_SOCKET_IFNAME="^docker0,lo"
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7

# Enable NCCL debugging (can be disabled for production)
export RAT_DEBUG_NCCL=1

# Setup scratch directory for Ray
USER_SCRATCH="/scratch/$USER"
echo "=== Directory Setup ==="
echo "User: $USER"
echo "Scratch directory: $USER_SCRATCH"
if [ -d "$USER_SCRATCH" ] && [ -w "$USER_SCRATCH" ]; then
    echo "✅ Scratch directory is available and writable"
    mkdir -p "$USER_SCRATCH/rat"
    echo "✅ Created RAT scratch directory: $USER_SCRATCH/rat"
else
    echo "⚠️  Scratch directory not available, Ray will use fallback"
fi
echo ""

# Ensure conda environment is activated
source /groups/cellmap/home/rhoadesj/micromamba/etc/profile.d/micromamba.sh
micromamba activate RAT

echo "=== Environment Check ==="
echo "Python executable: $(which python)"
echo "Python version: $(python --version)"
echo "CUDA devices visible: $CUDA_VISIBLE_DEVICES"
echo "Available GPUs: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo ""

# Create results directory
mkdir -p results/lsf_logs

# Change to experiments directory
cd /nrs/cellmap/rhoadesj/resolution-aware-transformer/experiments

echo "=== Starting RAT Training ==="
echo "Command: python run_experiment.py --config configs/medical_segmentation.yaml --num-gpus 8"
echo "Started at: $(date)"
echo ""

# Run the experiment with proper error handling and fallback
echo "Attempting distributed training with Ray Train..."
python ray_train.py --config configs/medical_segmentation.yaml --num-gpus 8

exit_code=$?

# If Ray Train fails with timeout or distributed errors, try simple runner as fallback
if [ $exit_code -ne 0 ]; then
    echo ""
    echo "⚠️  Ray Train failed with exit code $exit_code"
    echo "🔄 Attempting fallback to simple experiment runner..."
    echo ""
    
    # Disable Ray Train debugging for fallback
    export RAT_DEBUG_NCCL=0
    
    # Run with simple experiment runner (still uses distributed training but without Ray)
    python run_experiment.py --config configs/medical_segmentation.yaml --num-gpus 8
    
    fallback_exit_code=$?
    
    if [ $fallback_exit_code -eq 0 ]; then
        echo "✅ Fallback experiment completed successfully"
        exit_code=0
    else
        echo "❌ Both Ray Train and fallback failed"
        echo "🚀 Trying proper Ray cluster approach..."
        
        # Try the proper Ray cluster approach
        export RAY_TMPDIR="/scratch/$(whoami)/ray_$(whoami)"
        mkdir -p "$RAY_TMPDIR"
        
        # Start a simple Ray cluster
        ray start --head --port=6379 --num-cpus=96 --num-gpus=8 --temp-dir="$RAY_TMPDIR" &
        sleep 10
        
        export RAY_ADDRESS="localhost:6379"
        export WORLD_GPUS=8
        
        python experiments/cluster_bootstrap.py --config configs/medical_segmentation.yaml --num-gpus 8
        ray_cluster_exit_code=$?
        
        ray stop
        
        if [ $ray_cluster_exit_code -eq 0 ]; then
            echo "✅ Ray cluster approach completed successfully"
            exit_code=0
        else
            echo "❌ All approaches failed"
            exit_code=$ray_cluster_exit_code
        fi
    fi
fi

echo ""
echo "=== Job Completed ==="
echo "Exit code: $exit_code"
echo "Finished at: $(date)"

exit $exit_code