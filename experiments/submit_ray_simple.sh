#!/bin/bash
#BSUB -J RAT_ray_simple
#BSUB -o results/lsf_logs/RAT_ray_simple_%J.out
#BSUB -e results/lsf_logs/RAT_ray_simple_%J.err
#BSUB -n 96
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4000,ngpus_excl_p=8]"
#BSUB -q gpu_h100
#BSUB -gpu "num=8"

# Simple Ray-based training script
# This avoids the complex multi-node Ray cluster setup

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
export NCCL_TIMEOUT_S=3600  # 1 hour timeout
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0
export NCCL_BUFFSIZE=8388608
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export NCCL_BLOCKING_WAIT=1

# Ray temporary directory
export RAY_TMPDIR="/scratch/$(whoami)/ray_$(whoami)"
mkdir -p "$RAY_TMPDIR"

echo "=== Starting Simple Ray Training ==="
echo "Ray temp dir: $RAY_TMPDIR"

# Start a simple local Ray cluster and run training
echo "Starting local Ray cluster..."

# Start Ray head locally
ray start --head --port=6379 --num-cpus=96 --num-gpus=8 --temp-dir="$RAY_TMPDIR" --disable-usage-stats &
sleep 10

# Set Ray address for the training script
export RAY_ADDRESS="localhost:6379"
export WORLD_GPUS=8

echo "Ray cluster started, beginning training..."

# Run the cluster bootstrap script
python experiments/cluster_bootstrap.py --config experiments/configs/medical_segmentation.yaml --num-gpus 8 --name "RAT_simple_ray"

exit_code=$?

echo ""
echo "=== Cleaning Up ==="
ray stop
echo "Ray cluster stopped"

echo ""
echo "=== Job Completed ==="
echo "Exit code: $exit_code"
echo "Finished at: $(date)"

exit $exit_code