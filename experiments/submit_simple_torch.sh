#!/bin/bash

#BSUB -J RAT_simple_torch
#BSUB -o results/lsf_logs/RAT_simple_torch_%J.out
#BSUB -e results/lsf_logs/RAT_simple_torch_%J.err
#BSUB -n 96
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4000,ngpus_excl_p=8]"
#BSUB -q gpu_h100
#BSUB -gpu "num=8"

echo "=== Simple PyTorch Distributed Training ==="
echo "Job ID: $LSB_JOBID"
echo "Started at: $(date)"

# Setup environment
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# NCCL settings
export NCCL_TIMEOUT_S=3600
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0

# Distributed training settings
export MASTER_ADDR=localhost
export MASTER_PORT=12355
export WORLD_SIZE=8
export RANK=0

echo "Starting simple distributed PyTorch training..."

# Use torchrun for simple distributed training
torchrun \
    --nproc_per_node=8 \
    --master_addr=localhost \
    --master_port=12355 \
    simple_train.py \
    --config configs/medical_segmentation_fixed.yaml

exit_code=$?

echo "=== Job Completed ==="
echo "Exit code: $exit_code"
echo "Finished at: $(date)"

exit $exit_code