#!/usr/bin/env python3
"""
Migration script for transitioning from old RAT experiment structure to unified framework.
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
import yaml
import json

def migrate_config(old_config_path: Path, output_dir: Path) -> Path:
    """
    Migrate old configuration to new unified format.
    
    Args:
        old_config_path: Path to old configuration file
        output_dir: Directory to save new configuration
    
    Returns:
        Path to new configuration file
    """
    with open(old_config_path, 'r') as f:
        old_config = yaml.safe_load(f)
    
    # Create new unified configuration
    new_config = {
        "experiment_name": old_config.get("experiment_name", "migrated_experiment"),
        "description": old_config.get("description", "Migrated from old configuration"),
        "seed": old_config.get("seed", 42),
    }
    
    # Determine task type
    if "segmentation" in str(old_config_path).lower() or "isic" in str(old_config_path).lower():
        new_config["task_type"] = "segmentation"
    elif "detection" in str(old_config_path).lower() or "coco" in str(old_config_path).lower():
        new_config["task_type"] = "detection"
    else:
        new_config["task_type"] = "segmentation"  # Default
    
    # Migrate model configuration
    old_model = old_config.get("model", {})
    new_config["model"] = {
        "name": old_model.get("name", "rat"),
        "spatial_dims": old_model.get("spatial_dims", 2),
        "input_features": old_model.get("input_features", 3),
        "feature_dims": old_model.get("feature_dims", 128),
        "num_blocks": old_model.get("num_blocks", 4),
        "num_heads": old_model.get("num_heads", 8),
        "attention_type": old_model.get("attention_type", "dense"),
        "multi_scale": old_model.get("multi_scale", False),
        "scales": old_model.get("scales", [256, 128, 64]),
        "learnable_rose": old_model.get("learnable_rose", True),
        "mlp_ratio": old_model.get("mlp_ratio", 4),
        "mlp_dropout": old_model.get("mlp_dropout", 0.1),
    }
    
    # Add detection-specific parameters if needed
    if new_config["task_type"] == "detection":
        new_config["model"].update({
            "num_classes": old_model.get("num_classes", 80),
            "num_queries": old_model.get("num_queries", 100),
            "bbox_loss_coef": old_model.get("bbox_loss_coef", 5.0),
            "giou_loss_coef": old_model.get("giou_loss_coef", 2.0),
            "class_loss_coef": old_model.get("class_loss_coef", 1.0),
        })
    
    # Migrate training configuration
    old_training = old_config.get("training", {})
    new_config["training"] = {
        "epochs": old_training.get("epochs", 100),
        "auto_batch_size": True,  # Enable auto batch size
        "learning_rate": old_training.get("learning_rate", 1e-4),
        "weight_decay": old_training.get("weight_decay", 0.01),
        "scheduler": old_training.get("scheduler", "cosine"),
        "grad_clip": old_training.get("grad_clip", 1.0),
        "mixed_precision": True,  # Enable by default
        "deepspeed": True,  # Enable DeepSpeed Stage 2
        "zero_stage": 2,
        "use_accelerate": True,  # Enable Accelerate
        "save_freq": old_training.get("save_freq", 10),
    }
    
    # Migrate loss configuration for segmentation
    if new_config["task_type"] == "segmentation":
        new_config["training"]["loss"] = old_training.get("loss", "combined")
    
    # Set target batch size based on old batch size
    old_batch_size = old_training.get("batch_size", 4)
    new_config["training"]["target_batch_size"] = old_batch_size * 8  # Assume 8 GPUs
    
    # Migrate data configuration
    old_data = old_config.get("data", {})
    new_config["data"] = {
        "data_dir": old_data.get("data_dir", "/path/to/dataset"),
        "image_size": old_data.get("image_size", 256),
        "auto_optimize": True,  # Enable auto-optimization
    }
    
    # Add dataset-specific parameters
    if new_config["task_type"] == "segmentation":
        new_config["data"]["dataset_name"] = "isic2018"
    elif new_config["task_type"] == "detection":
        new_config["data"].update({
            "dataset_name": "coco",
            "train_split": old_data.get("train_split", "train2017"),
            "val_split": old_data.get("val_split", "val2017"),
        })
    
    # Migrate logging configuration
    old_logging = old_config.get("logging", {})
    new_config["logging"] = {
        "backend": old_logging.get("backend", "tensorboard"),
        "log_dir": old_logging.get("log_dir", "results/tensorboard_logs"),
        "use_mlflow": True,  # Enable MLFlow tracking
    }
    
    # Add distributed configuration
    new_config["distributed"] = {
        "auto_detect": True,
        "backend": "nccl",
    }
    
    # Add debug mode
    new_config["debug"] = old_config.get("debug", False)
    
    # Save new configuration
    new_config_name = f"unified_{old_config_path.stem}.yaml"
    new_config_path = output_dir / new_config_name
    
    with open(new_config_path, 'w') as f:
        yaml.dump(new_config, f, default_flow_style=False, indent=2)
    
    return new_config_path


def create_migration_script(old_experiment_dir: Path, output_dir: Path) -> Path:
    """
    Create a script to run the migrated experiment.
    
    Args:
        old_experiment_dir: Directory of old experiment
        output_dir: Directory to save migration script
    
    Returns:
        Path to migration script
    """
    # Determine experiment type
    if "segmentation" in str(old_experiment_dir).lower():
        task_type = "segmentation"
        data_dir_default = "/path/to/ISIC2018"
    elif "detection" in str(old_experiment_dir).lower():
        task_type = "detection"
        data_dir_default = "/path/to/COCO"
    else:
        task_type = "segmentation"
        data_dir_default = "/path/to/dataset"
    
    script_content = f"""#!/bin/bash
# Migrated training script for {old_experiment_dir.name}
# Generated by RAT migration tool

set -e  # Exit on error

# Configuration
CONFIG_FILE="configs/unified_{old_experiment_dir.name}.yaml"
DATA_DIR="{data_dir_default}"  # Update this path
OUTPUT_DIR="./outputs/{old_experiment_dir.name}_migrated"
TASK_TYPE="{task_type}"

echo "=== Migrated RAT Experiment: {old_experiment_dir.name} ==="
echo "Running unified training framework"
echo ""

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found: $CONFIG_FILE"
    echo "Make sure you've run the migration script first."
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Configuration:"
echo "  Config file: $CONFIG_FILE"
echo "  Task type: $TASK_TYPE"
echo "  Data directory: $DATA_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo ""

# Check for data directory
if [ ! -d "$DATA_DIR" ]; then
    echo "Warning: Data directory not found: $DATA_DIR"
    echo "Please update DATA_DIR in this script or use --data-dir"
    echo "Using debug mode for demonstration..."
    DEBUG_FLAG="--debug"
else
    DEBUG_FLAG=""
fi

# Run unified training
echo "Starting training with unified framework..."
python unified_train.py \\
    --config "$CONFIG_FILE" \\
    --task-type "$TASK_TYPE" \\
    --data-dir "$DATA_DIR" \\
    --output-dir "$OUTPUT_DIR" \\
    $DEBUG_FLAG

echo ""
echo "Training completed!"
echo "Results saved to: $OUTPUT_DIR"
echo ""

# Migration notes
echo "=== Migration Notes ==="
echo "This experiment has been migrated to the unified framework with the following improvements:"
echo "  - Automatic batch size optimization based on GPU memory"
echo "  - DeepSpeed Stage 2 enabled for better memory efficiency"
echo "  - HuggingFace Accelerate integration for simplified distributed training"
echo "  - MLFlow experiment tracking enabled"
echo "  - Auto-detection of distributed training environment"
echo ""
echo "To customize further:"
echo "  - Edit $CONFIG_FILE to adjust parameters"
echo "  - Use --batch-size, --learning-rate, --epochs to override config"
echo "  - Add --resume /path/to/checkpoint.pth to resume training"
echo ""
echo "For cluster submission:"
echo "  python cluster_launcher.py --config $CONFIG_FILE --experiment-type $TASK_TYPE --num-gpus 8"
"""
    
    script_path = output_dir / f"run_{old_experiment_dir.name}_migrated.sh"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make script executable
    script_path.chmod(0o755)
    
    return script_path


def main():
    parser = argparse.ArgumentParser(description="Migrate RAT experiments to unified framework")
    
    parser.add_argument("--old-experiments-dir", type=str, default=".",
                       help="Directory containing old experiments")
    parser.add_argument("--output-dir", type=str, default="./migrated",
                       help="Output directory for migrated configurations")
    parser.add_argument("--experiment", type=str, 
                       help="Specific experiment to migrate (default: migrate all)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be migrated without making changes")
    
    args = parser.parse_args()
    
    old_experiments_dir = Path(args.old_experiments_dir)
    output_dir = Path(args.output_dir)
    
    if not old_experiments_dir.exists():
        print(f"Error: Old experiments directory not found: {old_experiments_dir}")
        sys.exit(1)
    
    print("=== RAT Experiment Migration Tool ===")
    print(f"Migrating from: {old_experiments_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Find experiment directories
    if args.experiment:
        experiment_dirs = [old_experiments_dir / args.experiment]
        if not experiment_dirs[0].exists():
            print(f"Error: Experiment directory not found: {experiment_dirs[0]}")
            sys.exit(1)
    else:
        # Find all experiment directories
        experiment_dirs = []
        for item in old_experiments_dir.iterdir():
            if item.is_dir() and any(
                config_dir.is_dir() and config_dir.name == "configs"
                for config_dir in item.iterdir()
            ):
                experiment_dirs.append(item)
        
        if not experiment_dirs:
            print("No experiment directories found in the specified path.")
            print("Looking for directories containing 'configs' subdirectories.")
            sys.exit(1)
    
    print(f"Found {len(experiment_dirs)} experiment(s) to migrate:")
    for exp_dir in experiment_dirs:
        print(f"  - {exp_dir.name}")
    print()
    
    if args.dry_run:
        print("DRY RUN MODE - No files will be created")
        print()
    
    # Create output directories
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        configs_dir = output_dir / "configs"
        configs_dir.mkdir(exist_ok=True)
    
    migration_summary = []
    
    # Migrate each experiment
    for exp_dir in experiment_dirs:
        print(f"=== Migrating {exp_dir.name} ===")
        
        # Find configuration files
        configs_dir = exp_dir / "configs"
        if not configs_dir.exists():
            print(f"  Warning: No configs directory found in {exp_dir}")
            continue
        
        config_files = list(configs_dir.glob("*.yaml"))
        if not config_files:
            print(f"  Warning: No YAML config files found in {configs_dir}")
            continue
        
        print(f"  Found {len(config_files)} configuration file(s)")
        
        migrated_configs = []
        
        for config_file in config_files:
            print(f"    Migrating: {config_file.name}")
            
            if not args.dry_run:
                try:
                    new_config_path = migrate_config(config_file, output_dir / "configs")
                    migrated_configs.append(new_config_path)
                    print(f"      → {new_config_path.name}")
                except Exception as e:
                    print(f"      Error: {e}")
                    continue
            else:
                print(f"      → unified_{config_file.stem}.yaml")
        
        # Create migration script
        if not args.dry_run and migrated_configs:
            try:
                script_path = create_migration_script(exp_dir, output_dir)
                print(f"  Created migration script: {script_path.name}")
            except Exception as e:
                print(f"  Error creating migration script: {e}")
        
        migration_summary.append({
            "experiment": exp_dir.name,
            "configs_migrated": len(migrated_configs) if not args.dry_run else len(config_files),
            "success": True,
        })
        
        print()
    
    # Print summary
    print("=== Migration Summary ===")
    total_configs = sum(item["configs_migrated"] for item in migration_summary)
    successful_experiments = sum(1 for item in migration_summary if item["success"])
    
    print(f"Experiments migrated: {successful_experiments}/{len(migration_summary)}")
    print(f"Total configs migrated: {total_configs}")
    print()
    
    if not args.dry_run:
        print("Migration completed!")
        print(f"Migrated files saved to: {output_dir}")
        print()
        print("Next steps:")
        print("1. Review and update the migrated configuration files")
        print("2. Update data directory paths in the configurations")
        print("3. Run the generated migration scripts to test")
        print("4. Use the unified training framework:")
        print(f"   python unified_train.py --config {output_dir}/configs/unified_*.yaml")
        print()
        print("For cluster submission:")
        print(f"   python cluster_launcher.py --config {output_dir}/configs/unified_*.yaml --experiment-type segmentation")
    else:
        print("Dry run completed. Use --no-dry-run to perform actual migration.")


if __name__ == "__main__":
    main()