#!/bin/bash
# Distributed Training Launcher for RAT Experiments

# Set environment variables
export MASTER_ADDR=localhost
export MASTER_PORT=12355
export WORLD_SIZE=${1:-8}  # Default to 8 GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Configuration
CONFIG_DIR=${2:-"medical_segmentation/configs"}
RESULTS_DIR=${3:-"/shared/results/rat_experiments"}
CHECKPOINT_DIR=${4:-"/shared/checkpoints/rat"}

echo "Starting distributed training with $WORLD_SIZE GPUs"
echo "Config directory: $CONFIG_DIR"
echo "Results directory: $RESULTS_DIR"
echo "Checkpoint directory: $CHECKPOINT_DIR"

# Create directories
mkdir -p "$RESULTS_DIR/tensorboard_logs"
mkdir -p "$CHECKPOINT_DIR"

# Run distributed training
torchrun \
    --nnodes=1 \
    --nproc_per_node=$WORLD_SIZE \
    --master_addr=localhost \
    --master_port=12355 \
    train_distributed.py \
    --config_dir "$CONFIG_DIR" \
    --results_dir "$RESULTS_DIR" \
    --checkpoint_dir "$CHECKPOINT_DIR"

echo "Training completed"
