#!/bin/bash
"""
Example: Local Training with Auto-Optimization
Demonstrates how to use the unified training framework for local development.
"""

set -e  # Exit on error

# Configuration
CONFIG_FILE="configs/unified_template.yaml"
DATA_DIR="/path/to/ISIC2018"  # Update this path
OUTPUT_DIR="./outputs/local_example"
EXPERIMENT_NAME="local_segmentation_example"

echo "=== RAT Unified Training Framework Example ==="
echo "Running local training with auto-optimization"
echo ""

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Warning: Data directory not found: $DATA_DIR"
    echo "Please update DATA_DIR in this script or use --data-dir"
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
echo "  Debug mode: ${DEBUG_FLAG:-disabled}"
echo ""

# Run training with auto-optimization
echo "Starting training..."
python unified_train.py \
    --config "$CONFIG_FILE" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --task-type segmentation \
    $DEBUG_FLAG

echo ""
echo "Training completed!"
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
echo "To view TensorBoard logs:"
echo "  tensorboard --logdir $OUTPUT_DIR/../tensorboard_logs"
echo ""
echo "To resume training:"
echo "  python unified_train.py --config $CONFIG_FILE --data-dir $DATA_DIR --output-dir $OUTPUT_DIR --resume $OUTPUT_DIR/best_model.pth"