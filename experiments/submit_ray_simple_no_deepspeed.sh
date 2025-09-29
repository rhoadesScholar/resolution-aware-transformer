#!/bin/bash

#BSUB -J RAT_ray_simple_no_ds
#BSUB -o results/lsf_logs/RAT_ray_simple_no_ds_%J.out
#BSUB -e results/lsf_logs/RAT_ray_simple_no_ds_%J.err
#BSUB -n 96
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4000,ngpus_excl_p=8]"
#BSUB -q gpu_h100
#BSUB -gpu "num=8"

# Simple Ray-based training script without DeepSpeed
# This avoids DeepSpeed initialization issues

echo "=== LSF Job Information ==="
echo "Job ID: $LSB_JOBID"
echo "Job Name: $LSB_JOBNAME"
echo "Queue: $LSB_QUEUE"
echo "Hosts: $LSB_HOSTS"
echo "Number of processors: $LSB_DJOB_NUMPROC"
echo "Processors on each host: $LSB_HOSTS"

echo "=== Environment Setup ==="
# Set optimal OMP settings for 8 GPUs on single node
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NCCL_ASYNC_ERROR_HANDLING=1

# Setup Ray environment
export RAY_TMPDIR="/scratch/$USER/ray_tmp"
export RAY_DEDUP_LOGS=0

# Create directories
mkdir -p /scratch/$USER/ray_tmp
mkdir -p results/lsf_logs

echo "Starting Ray cluster..."

# Start Ray head node
ray start --head \
    --port=6379 \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265 \
    --temp-dir=/scratch/$USER/ray_tmp \
    --num-gpus=8 \
    --num-cpus=96

# Wait for Ray to fully initialize
echo "Waiting for Ray cluster to stabilize..."
sleep 5

# Run training WITHOUT DeepSpeed
echo "=== Starting Training (No DeepSpeed) ==="
python cluster_bootstrap.py \
    --config configs/medical_segmentation.yaml \
    --disable-deepspeed \
    --num-gpus 8 \
    --name "RAT_ray_simple_no_ds"

exit_code=$?

echo "=== Cleanup ==="
ray stop --force
sleep 2

echo "=== Job Completed ==="
echo "Exit code: $exit_code"
echo "Finished at: $(date)"

exit $exit_code