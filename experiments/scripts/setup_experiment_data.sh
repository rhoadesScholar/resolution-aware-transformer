#!/bin/bash
# Shared data setup script for RAT experiments
# This script is sourced by individual experiment scripts to set up data

# This script should be sourced, not executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Error: This script should be sourced, not executed directly"
    echo "Usage: . experiments/scripts/setup_experiment_data.sh"
    exit 1
fi

echo "Setting up experiment data..."

# Read configuration to determine data sources
DATASET_CONFIG=$(python3 experiments/config_manager.py --dump)
USE_SAMPLE_DATA=$(echo "$DATASET_CONFIG" | grep "use_sample_data" | cut -d'=' -f2 | xargs)
ISIC_SOURCE=$(echo "$DATASET_CONFIG" | grep "isic_source_dir" | cut -d'=' -f2 | xargs)
COCO_SOURCE=$(echo "$DATASET_CONFIG" | grep "coco_source_dir" | cut -d'=' -f2 | xargs)
ISIC_SAMPLE_SIZE=$(echo "$DATASET_CONFIG" | grep "isic_sample_size" | cut -d'=' -f2 | xargs)
COCO_SAMPLE_SIZE=$(echo "$DATASET_CONFIG" | grep "coco_sample_size" | cut -d'=' -f2 | xargs)

# ISIC dataset setup
if [ "$USE_SAMPLE_DATA" = "true" ] || [ -z "$ISIC_SOURCE" ]; then
    echo "Creating sample ISIC2018 dataset..."
    python experiments/scripts/setup_isic.py \
        --sample_only \
        --output_dir "$LOCAL_DATA_DIR/sample_ISIC2018" \
        --num_samples ${ISIC_SAMPLE_SIZE:-100}
else
    if [ -d "$ISIC_SOURCE" ]; then
        echo "Copying ISIC2018 from $ISIC_SOURCE to local storage..."
        rsync -av "$ISIC_SOURCE/" "$LOCAL_DATA_DIR/ISIC2018/"
    else
        echo "Warning: ISIC source directory not found: $ISIC_SOURCE"
        echo "Falling back to sample dataset..."
        python experiments/scripts/setup_isic.py \
            --sample_only \
            --output_dir "$LOCAL_DATA_DIR/sample_ISIC2018" \
            --num_samples ${ISIC_SAMPLE_SIZE:-100}
    fi
fi

# COCO dataset setup
if [ "$USE_SAMPLE_DATA" = "true" ] || [ -z "$COCO_SOURCE" ]; then
    echo "Creating sample COCO2017 dataset..."
    python experiments/scripts/setup_coco.py \
        --sample_only \
        --output_dir "$LOCAL_DATA_DIR/sample_COCO2017" \
        --num_samples ${COCO_SAMPLE_SIZE:-50}
else
    if [ -d "$COCO_SOURCE" ]; then
        echo "Copying COCO2017 from $COCO_SOURCE to local storage..."
        rsync -av "$COCO_SOURCE/" "$LOCAL_DATA_DIR/COCO2017/"
    else
        echo "Warning: COCO source directory not found: $COCO_SOURCE"
        echo "Falling back to sample dataset..."
        python experiments/scripts/setup_coco.py \
            --sample_only \
            --output_dir "$LOCAL_DATA_DIR/sample_COCO2017" \
            --num_samples ${COCO_SAMPLE_SIZE:-50}
    fi
fi

echo "Data setup completed. Local data directory contents:"
du -sh "$LOCAL_DATA_DIR"/* 2>/dev/null || echo "No data directories found"

# Update cluster configurations
echo "Updating configurations for cluster..."
python scripts/update_cluster_configs.py \
    --data_dir "$LOCAL_DATA_DIR" \
    --results_dir "$NETWORK_RESULTS_DIR" \
    --checkpoint_dir "$NETWORK_CHECKPOINTS_DIR" \
    --num_gpus "$NUM_GPUS"

echo "Data setup and configuration update completed."