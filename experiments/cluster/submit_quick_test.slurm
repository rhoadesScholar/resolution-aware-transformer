#!/bin/bash
#SBATCH --job-name=rat_quick_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --output=/shared/results/rat_experiments/slurm_logs/rat_quick_%j.out
#SBATCH --error=/shared/results/rat_experiments/slurm_logs/rat_quick_%j.err

# RAT Quick Test - 2 GPU test run for validation

set -e

echo "======================================================"
echo "RAT Quick Test - 2 GPU Validation"
echo "Job ID: $SLURM_JOB_ID"
echo "======================================================"

# Configuration
LOCAL_DATA_DIR="/tmp/rat_test_$SLURM_JOB_ID"
NETWORK_RESULTS_DIR="/shared/results/rat_experiments"
NETWORK_CHECKPOINTS_DIR="/shared/checkpoints/rat"
NUM_GPUS=2

# Environment setup
source ~/.bashrc
# conda activate rat_env  # Uncomment for your environment

# Create directories
mkdir -p "$LOCAL_DATA_DIR"
mkdir -p "$NETWORK_RESULTS_DIR/slurm_logs"
mkdir -p "$NETWORK_RESULTS_DIR/tensorboard_logs"
mkdir -p "$NETWORK_CHECKPOINTS_DIR"

# Create small sample datasets
echo "Creating sample datasets for quick test..."
python scripts/setup_isic.py \\
    --sample_only \\
    --output_dir "$LOCAL_DATA_DIR/sample_ISIC2018" \\
    --num_samples 20

python scripts/setup_coco.py \\
    --sample_only \\
    --output_dir "$LOCAL_DATA_DIR/sample_COCO2017" \\
    --num_samples 10

# Update configurations
echo "Updating configurations..."
python scripts/update_cluster_configs.py \\
    --data_dir "$LOCAL_DATA_DIR" \\
    --results_dir "$NETWORK_RESULTS_DIR" \\
    --checkpoint_dir "$NETWORK_CHECKPOINTS_DIR" \\
    --num_gpus "$NUM_GPUS"

# Create quick test config
cat > "experiments/configs/quick_test.yaml" << EOF
experiment_name: "quick_test_2gpu"
description: "Quick validation test with 2 GPUs"

data:
  data_dir: "$LOCAL_DATA_DIR/sample_ISIC2018"
  image_size: 128
  batch_size: 4
  num_workers: 4

model:
  name: "simple_test"

training:
  epochs: 2
  learning_rate: 0.001
  mixed_precision: true

logging:
  backend: "tensorboard"
  log_dir: "$NETWORK_RESULTS_DIR/tensorboard_logs"

cluster:
  num_gpus: $NUM_GPUS
  distributed_backend: "nccl"
EOF

# Set environment variables
export MASTER_ADDR=localhost
export MASTER_PORT=12356
export WORLD_SIZE=$NUM_GPUS
export CUDA_VISIBLE_DEVICES=0,1

echo "Starting quick test with $NUM_GPUS GPUs..."

# Run quick test
torchrun \\
    --nnodes=1 \\
    --nproc_per_node=$NUM_GPUS \\
    --master_addr=$MASTER_ADDR \\
    --master_port=$MASTER_PORT \\
    experiments/train_distributed.py \\
    --config_dir experiments/configs \\
    --results_dir "$NETWORK_RESULTS_DIR" \\
    --checkpoint_dir "$NETWORK_CHECKPOINTS_DIR"

echo "Quick test completed!"

# Save test results
TEST_SUMMARY="$NETWORK_RESULTS_DIR/quick_test_summary_$SLURM_JOB_ID.txt"
cat > "$TEST_SUMMARY" << EOF
RAT Quick Test Summary
=====================
Job ID: $SLURM_JOB_ID
GPUs: $NUM_GPUS
Duration: $(date)
Status: SUCCESS

GPU Information:
$(nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv)

Test Configuration:
- Sample ISIC dataset: 20 images
- Sample COCO dataset: 10 images
- Training epochs: 2
- Batch size: 4 per GPU
- Image size: 128x128

Next Steps:
1. Check logs in: $NETWORK_RESULTS_DIR/slurm_logs/
2. View TensorBoard: tensorboard --logdir $NETWORK_RESULTS_DIR/tensorboard_logs
3. Submit full job: sbatch cluster_scripts/submit_experiments.slurm
EOF

echo "Test summary saved to: $TEST_SUMMARY"

# Cleanup
rm -rf "$LOCAL_DATA_DIR"

echo "======================================================"
echo "Quick test completed successfully!"
echo "Check results in: $NETWORK_RESULTS_DIR"
echo "======================================================"