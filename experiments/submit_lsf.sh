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

# Run the experiment with proper error handling
python run_experiment.py --config configs/medical_segmentation.yaml --num-gpus 8

exit_code=$?

echo ""
echo "=== Job Completed ==="
echo "Exit code: $exit_code"
echo "Finished at: $(date)"

exit $exit_code