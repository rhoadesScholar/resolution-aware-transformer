#!/bin/bash
"""
Example: Multi-GPU Training with Accelerate
Demonstrates distributed training using HuggingFace Accelerate.
"""

set -e  # Exit on error

# Configuration
CONFIG_FILE="configs/unified_template.yaml"
DATA_DIR="/path/to/ISIC2018"  # Update this path
OUTPUT_DIR="./outputs/multi_gpu_example"
NUM_GPUS=4

echo "=== Multi-GPU Training with Accelerate Example ==="
echo "Running distributed training on $NUM_GPUS GPUs"
echo ""

# Check GPU availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi not found. This example requires CUDA GPUs."
    exit 1
fi

AVAILABLE_GPUS=$(nvidia-smi -L | wc -l)
if [ $AVAILABLE_GPUS -lt $NUM_GPUS ]; then
    echo "Warning: Only $AVAILABLE_GPUS GPUs available, but $NUM_GPUS requested."
    echo "Adjusting to use $AVAILABLE_GPUS GPUs..."
    NUM_GPUS=$AVAILABLE_GPUS
fi

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Warning: Data directory not found: $DATA_DIR"
    echo "Using debug mode for demonstration..."
    DEBUG_FLAG="--debug"
else
    DEBUG_FLAG=""
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Configuration:"
echo "  Config file: $CONFIG_FILE"
echo "  Data directory: $DATA_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo "  Number of GPUs: $NUM_GPUS"
echo "  Debug mode: ${DEBUG_FLAG:-disabled}"
echo ""

# Method 1: Using Accelerate (Recommended)
if command -v accelerate &> /dev/null; then
    echo "Using HuggingFace Accelerate for distributed training..."
    
    # Configure accelerate if not already done
    if [ ! -f ~/.cache/huggingface/accelerate/default_config.yaml ]; then
        echo "Configuring Accelerate..."
        accelerate config default
    fi
    
    # Launch with Accelerate
    accelerate launch \
        --num_processes=$NUM_GPUS \
        --mixed_precision=fp16 \
        unified_train.py \
        --config "$CONFIG_FILE" \
        --data-dir "$DATA_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --task-type segmentation \
        $DEBUG_FLAG

# Method 2: Using torchrun (Fallback)
else
    echo "Accelerate not available, using torchrun..."
    
    # Set environment variables
    export MASTER_ADDR=localhost
    export MASTER_PORT=12355
    
    # Launch with torchrun
    python -m torch.distributed.run \
        --nproc_per_node=$NUM_GPUS \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        unified_train.py \
        --config "$CONFIG_FILE" \
        --data-dir "$DATA_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --task-type segmentation \
        $DEBUG_FLAG
fi

echo ""
echo "Multi-GPU training completed!"
echo "Results saved to: $OUTPUT_DIR"
echo ""

# Show results
if [ -f "$OUTPUT_DIR/training_results.json" ]; then
    echo "=== Training Results ==="
    python -c "
import json
with open('$OUTPUT_DIR/training_results.json', 'r') as f:
    results = json.load(f)
print(f'Best metric: {results.get(\"best_metric\", \"N/A\")}')
print(f'Total time: {results.get(\"total_time\", \"N/A\")} seconds')
print(f'Total epochs: {len(results.get(\"training_stats\", []))}')
"
fi

echo ""
echo "Performance tips:"
echo "  - The unified framework automatically optimizes batch sizes for multi-GPU"
echo "  - DeepSpeed Stage 2 is auto-enabled for better memory efficiency"
echo "  - Mixed precision is auto-enabled for Ampere+ GPUs"
echo ""
echo "To monitor GPU usage during training:"
echo "  watch -n 1 nvidia-smi"