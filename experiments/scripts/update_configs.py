#!/usr/bin/env python3
"""
Configuration Update Script

Updates experiment configuration files with correct dataset paths after data setup.

Usage:
    python update_configs.py --data_dir /path/to/data
    python update_configs.py --isic_dir /path/to/ISIC2018 --coco_dir /path/to/COCO2017
"""

import argparse
import yaml
from pathlib import Path
from typing import Dict, Any


def update_yaml_file(config_path: Path, updates: Dict[str, Any]):
    """Update a YAML configuration file with new values."""
    print(f"Updating {config_path.name}...")

    # Load existing config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Apply updates recursively
    def update_nested_dict(d: dict, updates: dict):
        for key, value in updates.items():
            if "." in key:
                # Handle nested keys like 'data.data_dir'
                keys = key.split(".")
                target = d
                for k in keys[:-1]:
                    if k not in target:
                        target[k] = {}
                    target = target[k]
                target[keys[-1]] = value
            else:
                d[key] = value

    update_nested_dict(config, updates)

    # Save updated config
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False, indent=2)

    print(f"  ✓ Updated {config_path}")


def update_medical_segmentation_configs(data_dir: str):
    """Update medical segmentation configuration files."""
    configs_dir = Path("experiments/medical_segmentation/configs")

    if not configs_dir.exists():
        print(f"Warning: {configs_dir} not found")
        return

    # Update all config files in the directory
    config_files = list(configs_dir.glob("*.yaml")) + list(configs_dir.glob("*.yml"))

    updates = {"data.data_dir": str(data_dir)}

    for config_file in config_files:
        try:
            update_yaml_file(config_file, updates)
        except Exception as e:
            print(f"Error updating {config_file}: {e}")


def update_object_detection_configs(data_dir: str):
    """Update object detection configuration files."""
    configs_dir = Path("experiments/object_detection/configs")

    if not configs_dir.exists():
        print(f"Warning: {configs_dir} not found")
        return

    # Update all config files in the directory
    config_files = list(configs_dir.glob("*.yaml")) + list(configs_dir.glob("*.yml"))

    updates = {"data.data_dir": str(data_dir)}

    for config_file in config_files:
        try:
            update_yaml_file(config_file, updates)
        except Exception as e:
            print(f"Error updating {config_file}: {e}")


def update_ablation_configs(data_dir: str):
    """Update ablation study configuration files."""
    configs_dir = Path("experiments/ablations/configs")

    if not configs_dir.exists():
        print(f"Warning: {configs_dir} not found")
        return

    config_files = list(configs_dir.glob("*.yaml")) + list(configs_dir.glob("*.yml"))

    updates = {"data.data_dir": str(data_dir)}

    for config_file in config_files:
        try:
            update_yaml_file(config_file, updates)
        except Exception as e:
            print(f"Error updating {config_file}: {e}")


def create_quick_test_configs(isic_dir: str, coco_dir: str):
    """Create quick test configurations for rapid experimentation."""

    # Quick ISIC config
    quick_isic_config = {
        "experiment_name": "quick_test_isic",
        "description": "Quick test configuration for ISIC dataset",
        "seed": 42,
        "data": {
            "data_dir": str(isic_dir),
            "image_size": 128,  # Smaller for faster training
            "num_workers": 2,
        },
        "model": {
            "name": "rat_single",
            "multi_scale": False,
            "num_blocks": 2,  # Smaller model
            "feature_dim": 128,
        },
        "training": {
            "epochs": 5,  # Few epochs for quick test
            "batch_size": 8,
            "learning_rate": 0.001,
            "weight_decay": 0.01,
            "scheduler": "cosine",
            "loss": "combined",
            "grad_clip": 1.0,
            "save_freq": 2,
        },
        "wandb": {"enabled": False},
        "debug": True,
    }

    # Quick COCO config
    quick_coco_config = {
        "experiment_name": "quick_test_coco",
        "description": "Quick test configuration for COCO dataset",
        "seed": 42,
        "data": {
            "data_dir": str(coco_dir),
            "image_size": 256,  # Smaller for faster training
            "num_workers": 2,
            "max_samples": 100,  # Limit samples for quick test
        },
        "model": {
            "name": "rat_single",
            "multi_scale": False,
            "num_blocks": 2,
            "feature_dim": 128,
        },
        "training": {
            "epochs": 3,
            "batch_size": 4,
            "learning_rate": 0.001,
            "weight_decay": 0.01,
            "scheduler": "cosine",
            "grad_clip": 1.0,
            "save_freq": 1,
        },
        "wandb": {"enabled": False},
        "debug": True,
    }

    # Save quick configs
    quick_configs_dir = Path("experiments/configs")
    quick_configs_dir.mkdir(exist_ok=True)

    with open(quick_configs_dir / "quick_test_isic.yaml", "w") as f:
        yaml.safe_dump(quick_isic_config, f, default_flow_style=False, indent=2)

    with open(quick_configs_dir / "quick_test_coco.yaml", "w") as f:
        yaml.safe_dump(quick_coco_config, f, default_flow_style=False, indent=2)

    print("Created quick test configurations:")
    print(f"  - {quick_configs_dir / 'quick_test_isic.yaml'}")
    print(f"  - {quick_configs_dir / 'quick_test_coco.yaml'}")


def verify_data_paths(isic_dir: str | None = None, coco_dir: str | None = None):
    """Verify that data directories exist and have expected structure."""
    print("Verifying data paths...")

    if isic_dir:
        isic_path = Path(isic_dir)
        if isic_path.exists():
            images_dir = isic_path / "images"
            masks_dir = isic_path / "masks"
            splits_file = isic_path / "splits.json"

            print(f"ISIC dataset at {isic_path}:")
            print(
                f"  ✓ Images: {len(list(images_dir.glob('*'))) if images_dir.exists() else 0}"
            )
            print(
                f"  ✓ Masks: {len(list(masks_dir.glob('*'))) if masks_dir.exists() else 0}"
            )
            print(f"  ✓ Splits: {'Yes' if splits_file.exists() else 'No'}")
        else:
            print(f"✗ ISIC directory not found: {isic_dir}")

    if coco_dir:
        coco_path = Path(coco_dir)
        if coco_path.exists():
            train_dir = coco_path / "train2017"
            val_dir = coco_path / "val2017"
            ann_dir = coco_path / "annotations"

            print(f"COCO dataset at {coco_path}:")
            print(
                f"  ✓ Train images: {len(list(train_dir.glob('*'))) if train_dir.exists() else 0}"
            )
            print(
                f"  ✓ Val images: {len(list(val_dir.glob('*'))) if val_dir.exists() else 0}"
            )
            print(
                f"  ✓ Annotations: {len(list(ann_dir.glob('*.json'))) if ann_dir.exists() else 0}"
            )
        else:
            print(f"✗ COCO directory not found: {coco_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Update experiment configuration files"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Base data directory (will look for ISIC2018 and COCO2017 subdirectories)",
    )
    parser.add_argument("--isic_dir", type=str, help="ISIC dataset directory")
    parser.add_argument("--coco_dir", type=str, help="COCO dataset directory")
    parser.add_argument(
        "--create_quick_configs",
        action="store_true",
        help="Create quick test configurations",
    )

    args = parser.parse_args()

    # Determine dataset directories
    isic_dir = args.isic_dir
    coco_dir = args.coco_dir

    if args.data_dir:
        base_dir = Path(args.data_dir)
        if not isic_dir:
            potential_isic = base_dir / "ISIC2018"
            if potential_isic.exists():
                isic_dir = str(potential_isic)

        if not coco_dir:
            potential_coco = base_dir / "COCO2017"
            if potential_coco.exists():
                coco_dir = str(potential_coco)

    # Verify paths
    verify_data_paths(isic_dir, coco_dir)

    # Update configs
    if isic_dir:
        update_medical_segmentation_configs(isic_dir)

    if coco_dir:
        update_object_detection_configs(coco_dir)

    # Update ablation configs (can use either dataset)
    ablation_data_dir = isic_dir or coco_dir
    if ablation_data_dir:
        update_ablation_configs(ablation_data_dir)

    # Create quick test configs if requested
    if args.create_quick_configs and (isic_dir or coco_dir):
        create_quick_test_configs(isic_dir or "", coco_dir or "")

    print("\nConfiguration update complete!")
    print("\nNext steps:")
    print("1. Run quick tests: python experiments/run_experiments.py --quick")
    print("2. Run full experiments: python experiments/run_experiments.py")


if __name__ == "__main__":
    main()
