#!/usr/bin/env python3
"""
Organize existing unstructured experiment results.
This script moves the current RAT_ray_train folder to a properly named location.
"""

import os
import shutil
from pathlib import Path
from datetime import datetime


def organize_existing_experiments():
    """Move existing unorganized experiments to proper structure."""
    results_dir = Path(
        "/nrs/cellmap/rhoadesj/resolution-aware-transformer/experiments/results/experiments"
    )

    # Find unorganized experiment directories (not in active/completed/failed/archived)
    unorganized_dirs = []
    for item in results_dir.iterdir():
        if item.is_dir() and item.name not in [
            "active",
            "completed",
            "failed",
            "archived",
        ]:
            unorganized_dirs.append(item)

    print(f"Found {len(unorganized_dirs)} unorganized experiment directories:")

    for exp_dir in unorganized_dirs:
        print(f"\nProcessing: {exp_dir.name}")

        # Try to determine experiment status
        result_json = exp_dir / "TorchTrainer_*" / "result.json"
        result_files = list(exp_dir.glob("*/result.json"))

        if result_files:
            # Check if experiment completed successfully
            try:
                import json

                with open(result_files[0], "r") as f:
                    result_data = json.load(f)

                # Determine if successful based on result content
                if (
                    "error" in result_data
                    or result_data.get("training_iteration", 0) == 0
                ):
                    status = "failed"
                else:
                    status = "completed"
            except:
                status = "failed"
        else:
            status = "failed"  # No result file means it didn't complete

        # Create descriptive name with timestamp
        if exp_dir.name == "RAT_ray_train":
            # Get creation time for timestamp
            creation_time = datetime.fromtimestamp(exp_dir.stat().st_ctime)
            timestamp = creation_time.strftime("%Y%m%d_%H%M%S")
            new_name = f"RAT_ray_train_medical_segmentation_8gpu_ds_{timestamp}"
        else:
            new_name = exp_dir.name

        # Move to appropriate status directory
        target_dir = results_dir / status / new_name

        print(f"  Status: {status}")
        print(f"  Moving to: {target_dir}")

        # Create target directory parent if needed
        target_dir.parent.mkdir(parents=True, exist_ok=True)

        # Move the directory
        try:
            shutil.move(str(exp_dir), str(target_dir))
            print(f"  ✓ Successfully moved to {status}/{new_name}")
        except Exception as e:
            print(f"  ✗ Failed to move: {e}")


if __name__ == "__main__":
    organize_existing_experiments()
