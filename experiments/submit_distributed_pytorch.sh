#!/bin/bash
#BSUB -J RAT_medseg_simple
#BSUB -o results/lsf_logs/RAT_medseg_simple_%J.out
#BSUB -e results/lsf_logs/RAT_medseg_simple_%J.err
#BSUB -n 96
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4000,ngpus_excl_p=8]"
#BSUB -q gpu_h100
#BSUB -gpu "num=8"

# LSF job submission script for Resolution Aware Transformer training
# Simplified version that uses run_experiment.py directly instead of Ray Train

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

# NCCL network configuration for cluster stability (more conservative settings)
export NCCL_TIMEOUT_S=3600  # 1 hour timeout (less aggressive than 2 hours)
export NCCL_SOCKET_TIMEOUT=3600
export NCCL_CONNECT_TIMEOUT=180  # 3 minutes for initial connection
export NCCL_NET_RETRY_COUNT=3   # Fewer retries to fail faster
export NCCL_SOCKET_IFNAME="^docker0,lo"
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1

# Disable Ray-specific debugging since we're not using Ray Train
export RAT_DEBUG_NCCL=0

# Setup scratch directory
USER_SCRATCH="/scratch/$USER"
echo "=== Directory Setup ==="
echo "User: $USER"
echo "Scratch directory: $USER_SCRATCH"
if [ -d "$USER_SCRATCH" ] && [ -w "$USER_SCRATCH" ]; then
    echo "✅ Scratch directory is available and writable"
    mkdir -p "$USER_SCRATCH/rat"
    echo "✅ Created RAT scratch directory: $USER_SCRATCH/rat"
    export TMPDIR="$USER_SCRATCH/rat"
else
    echo "⚠️  Scratch directory not available, using default temp directory"
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
echo "NCCL timeout: ${NCCL_TIMEOUT_S}s"
echo "Temp directory: ${TMPDIR:-'system default'}"
echo ""

# Create results directory
mkdir -p results/lsf_logs

# Change to experiments directory
cd /nrs/cellmap/rhoadesj/resolution-aware-transformer/experiments

echo "=== Starting RAT Training (Simple Runner) ==="
echo "Command: python run_experiment.py --config configs/medical_segmentation.yaml --num-gpus 8"
echo "Started at: $(date)"
echo ""

# Run the experiment with simple runner (avoids Ray Train complexity)
python run_experiment.py --config configs/medical_segmentation.yaml --num-gpus 8

exit_code=$?

echo ""
echo "=== Job Completed ==="
echo "Exit code: $exit_code"
echo "Finished at: $(date)"

if [ $exit_code -eq 0 ]; then
    echo "✅ Training completed successfully"
else
    echo "❌ Training failed with exit code $exit_code"
    echo "Check the error logs above for details"
fi

exit $exit_code