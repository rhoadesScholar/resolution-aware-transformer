#!/bin/bash
#SBATCH --job-name=rat_experiments
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#SBATCH --mem=0
#SBATCH --output=/shared/results/rat_experiments/slurm_logs/rat_%j.out
#SBATCH --error=/shared/results/rat_experiments/slurm_logs/rat_%j.err

# RAT Experiments Cluster Submission Script
# This script sets up and runs RAT experiments on a multi-GPU cluster node

set -e  # Exit on any error

echo "======================================================"
echo "RAT Experiments - Node Setup and Training"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODEID"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "======================================================"

# Configuration
LOCAL_DATA_DIR="/tmp/rat_data_$SLURM_JOB_ID"
NETWORK_RESULTS_DIR="/shared/results/rat_experiments"
NETWORK_CHECKPOINTS_DIR="/shared/checkpoints/rat"
NUM_GPUS=8

# Environment setup
echo "Setting up environment..."
source ~/.bashrc
# Uncomment and modify the next line for your environment
# conda activate rat_env

# Load required modules (modify for your cluster)
# module load cuda/11.8
# module load python/3.10

echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "CUDA devices: $(python -c 'import torch; print(torch.cuda.device_count())')"

# Create necessary directories
echo "Creating directories..."
mkdir -p "$LOCAL_DATA_DIR"
mkdir -p "$NETWORK_RESULTS_DIR/slurm_logs"
mkdir -p "$NETWORK_RESULTS_DIR/tensorboard_logs"
mkdir -p "$NETWORK_CHECKPOINTS_DIR"

# Setup local data (fast I/O)
echo "Setting up local data..."
if [ -d "/shared/datasets/ISIC2018" ]; then
    echo "Copying ISIC2018 to local storage..."
    rsync -av "/shared/datasets/ISIC2018/" "$LOCAL_DATA_DIR/ISIC2018/"
else
    echo "Creating sample ISIC2018 dataset..."
    python scripts/setup_isic.py \
        --sample_only \
        --output_dir "$LOCAL_DATA_DIR/sample_ISIC2018" \
        --num_samples 100
fi

if [ -d "/shared/datasets/COCO2017" ]; then
    echo "Copying COCO2017 to local storage..."
    rsync -av "/shared/datasets/COCO2017/" "$LOCAL_DATA_DIR/COCO2017/"
else
    echo "Creating sample COCO2017 dataset..."
    python scripts/setup_coco.py \
        --sample_only \
        --output_dir "$LOCAL_DATA_DIR/sample_COCO2017" \
        --num_samples 50
fi

echo "Local data setup completed. Directory size:"
du -sh "$LOCAL_DATA_DIR"

# Update configurations for cluster
echo "Updating configurations for cluster..."
python scripts/update_cluster_configs.py \
    --data_dir "$LOCAL_DATA_DIR" \
    --results_dir "$NETWORK_RESULTS_DIR" \
    --checkpoint_dir "$NETWORK_CHECKPOINTS_DIR" \
    --num_gpus "$NUM_GPUS"

# Set distributed training environment variables
export MASTER_ADDR=localhost
export MASTER_PORT=12355
export WORLD_SIZE=$NUM_GPUS
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

echo "Starting distributed training..."
echo "Master address: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "World size: $WORLD_SIZE"

# Run medical segmentation experiments
echo "Running medical segmentation experiments..."
torchrun \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    experiments/train_distributed.py \
    --config_dir experiments/medical_segmentation/configs \
    --results_dir "$NETWORK_RESULTS_DIR" \
    --checkpoint_dir "$NETWORK_CHECKPOINTS_DIR"

# Run object detection experiments (if configs exist)
if [ -d "experiments/object_detection/configs" ]; then
    echo "Running object detection experiments..."
    torchrun \
        --nnodes=1 \
        --nproc_per_node=$NUM_GPUS \
        --master_addr=$MASTER_ADDR \
        --master_port=$((MASTER_PORT + 1)) \
        experiments/train_distributed.py \
        --config_dir experiments/object_detection/configs \
        --results_dir "$NETWORK_RESULTS_DIR" \
        --checkpoint_dir "$NETWORK_CHECKPOINTS_DIR"
fi

# Run ablation studies
if [ -d "experiments/ablations/configs" ]; then
    echo "Running ablation studies..."
    torchrun \
        --nnodes=1 \
        --nproc_per_node=$NUM_GPUS \
        --master_addr=$MASTER_ADDR \
        --master_port=$((MASTER_PORT + 2)) \
        experiments/train_distributed.py \
        --config_dir experiments/ablations/configs \
        --results_dir "$NETWORK_RESULTS_DIR" \
        --checkpoint_dir "$NETWORK_CHECKPOINTS_DIR"
fi

echo "Training completed successfully!"

# Save experiment summary
echo "Saving experiment summary..."
SUMMARY_FILE="$NETWORK_RESULTS_DIR/experiment_summary_$SLURM_JOB_ID.txt"
cat > "$SUMMARY_FILE" << EOF
RAT Experiments Summary
======================
Job ID: $SLURM_JOB_ID
Node: $SLURM_NODEID
Start Time: $(date)
GPUs Used: $NUM_GPUS
Local Data Dir: $LOCAL_DATA_DIR
Results Dir: $NETWORK_RESULTS_DIR
Checkpoints Dir: $NETWORK_CHECKPOINTS_DIR

Datasets:
$(ls -la "$LOCAL_DATA_DIR")

Configurations Used:
$(find experiments -name "cluster_*.yaml" | head -10)

GPU Memory Usage:
$(nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv)
EOF

echo "Experiment summary saved to: $SUMMARY_FILE"

# Optional: Clean up local data to free space
echo "Cleaning up local data..."
rm -rf "$LOCAL_DATA_DIR"

echo "======================================================"
echo "RAT Experiments Completed Successfully!"
echo "Results: $NETWORK_RESULTS_DIR"
echo "Checkpoints: $NETWORK_CHECKPOINTS_DIR"
echo "TensorBoard: tensorboard --logdir $NETWORK_RESULTS_DIR/tensorboard_logs"
echo "======================================================"