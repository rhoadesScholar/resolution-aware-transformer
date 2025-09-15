#!/usr/bin/env python3
"""
Cluster Setup Script for Resolution Aware Transformer Experiments

This script sets up RAT experiments for efficient execution on compute clusters
with multi-GPU nodes. It handles:
- Local data storage on compute nodes
- Network storage for results and checkpoints
- Multi-GPU distributed training configuration
- TensorBoard logging instead of WandB

Usage:
    # Setup for cluster with 8 GPUs
    python setup_cluster.py \
        --local_data_dir /tmp/rat_data \
        --network_results_dir /shared/results/rat_experiments \
        --network_checkpoints_dir /shared/checkpoints/rat \
        --num_gpus 8 \
        --node_id $SLURM_NODEID

    # Quick test setup
    python setup_cluster.py --quick --num_gpus 2
"""

# Import standard libraries
import argparse
import json
import os
import shutil
import subprocess
import sys

# Import third-party libraries
import yaml

# Import typing and pathlib
from pathlib import Path
from typing import Dict, Any


class ClusterSetup:
    """Setup RAT experiments for cluster computing."""

    def __init__(
        self,
        local_data_dir: str,
        network_results_dir: str,
        network_checkpoints_dir: str,
        num_gpus: int = 8,
        node_id: str = "0",
        quick: bool = False,
    ):
        self.local_data_dir = Path(local_data_dir)
        self.network_results_dir = Path(network_results_dir)
        self.network_checkpoints_dir = Path(network_checkpoints_dir)
        self.num_gpus = num_gpus
        self.node_id = node_id
        self.quick = quick

        # Create directories
        self.local_data_dir.mkdir(parents=True, exist_ok=True)
        self.network_results_dir.mkdir(parents=True, exist_ok=True)
        self.network_checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def setup_data_locally(self):
        """Set up datasets on local node storage for fast I/O."""
        print("Setting up datasets on local node storage...")

        if self.quick:
            # Create small sample datasets locally
            self._create_sample_datasets()
        else:
            # Check if we need to copy data from network storage
            network_data_sources = [
                "/shared/datasets/ISIC2018",
                "/shared/datasets/COCO2017",
            ]

            for source in network_data_sources:
                source_path = Path(source)
                if source_path.exists():
                    dataset_name = source_path.name
                    local_dataset_dir = self.local_data_dir / dataset_name

                    if not local_dataset_dir.exists():
                        print(f"Copying {dataset_name} to local storage...")
                        self._copy_with_progress(source_path, local_dataset_dir)
                    else:
                        print(f"{dataset_name} already exists locally")

    def _create_sample_datasets(self):
        """Create small sample datasets for testing."""
        print("Creating sample datasets for quick testing...")

        # Create sample ISIC dataset
        sample_isic_dir = self.local_data_dir / "sample_ISIC2018"
        if not sample_isic_dir.exists():
            cmd = [
                "python",
                "scripts/setup_isic.py",
                "--sample_only",
                "--output_dir",
                str(sample_isic_dir),
                "--num_samples",
                "20",
            ]
            subprocess.run(cmd, check=True)

        # Create sample COCO dataset
        sample_coco_dir = self.local_data_dir / "sample_COCO2017"
        if not sample_coco_dir.exists():
            cmd = [
                "python",
                "scripts/setup_coco.py",
                "--sample_only",
                "--output_dir",
                str(sample_coco_dir),
                "--num_samples",
                "10",
            ]
            subprocess.run(cmd, check=True)

    def _copy_with_progress(self, source: Path, dest: Path):
        """Copy directory with progress indication."""
        import time

        start_time = time.time()

        # Use rsync for efficient copying with progress
        cmd = ["rsync", "-avh", "--progress", str(source) + "/", str(dest) + "/"]

        try:
            subprocess.run(cmd, check=True)
            elapsed = time.time() - start_time
            print(f"Copy completed in {elapsed:.1f} seconds")
        except subprocess.CalledProcessError:
            # Fallback to Python copy
            print("rsync failed, using Python copy...")
            shutil.copytree(source, dest, dirs_exist_ok=True)

    def create_cluster_configs(self):
        """Create cluster-optimized configuration files."""
        print("Creating cluster-optimized configurations...")

        # Base cluster configuration
        cluster_config = {
            "cluster": {
                "num_gpus": self.num_gpus,
                "node_id": self.node_id,
                "distributed_backend": "nccl",
                "find_unused_parameters": True,
                "gradient_checkpointing": True,
            },
            "data": {
                "num_workers": min(16, self.num_gpus * 2),  # 2 workers per GPU
                "pin_memory": True,
                "prefetch_factor": 2,
                "persistent_workers": True,
            },
            "training": {
                "mixed_precision": True,
                "compile_model": True,  # PyTorch 2.0 compilation
                "gradient_accumulation_steps": max(1, 32 // self.num_gpus),
                "save_freq": 5 if self.quick else 10,
                "eval_freq": 2 if self.quick else 5,
            },
            "logging": {
                "backend": "tensorboard",
                "log_dir": str(self.network_results_dir / "tensorboard_logs"),
                "log_freq": 50,
                "save_predictions": True,
                "save_attention_maps": False,  # Disable for cluster efficiency
            },
            "checkpoints": {
                "save_dir": str(self.network_checkpoints_dir),
                "save_top_k": 3,
                "save_last": True,
                "monitor": "val_dice_score",
                "mode": "max",
            },
        }

        # Update experiment configs
        self._update_medical_segmentation_configs(cluster_config)
        self._update_object_detection_configs(cluster_config)
        self._update_ablation_configs(cluster_config)

        # Create distributed training wrapper configs
        self._create_distributed_training_configs()

    def _update_medical_segmentation_configs(self, cluster_config: Dict[str, Any]):
        """Update medical segmentation configs for cluster."""
        configs_dir = Path("experiments/medical_segmentation/configs")

        for config_file in configs_dir.glob("*.yaml"):
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)

            # Update data paths
            if self.quick:
                config["data"]["data_dir"] = str(
                    self.local_data_dir / "sample_ISIC2018"
                )
            else:
                config["data"]["data_dir"] = str(self.local_data_dir / "ISIC2018")

            # Update for cluster
            config.update(cluster_config)

            # Adjust batch size for multi-GPU
            original_batch_size = config["training"].get("batch_size", 16)
            config["training"]["batch_size"] = max(
                1, original_batch_size // self.num_gpus
            )
            config["training"]["effective_batch_size"] = original_batch_size

            # Adjust epochs for quick mode
            if self.quick:
                config["training"]["epochs"] = 3

            # Remove wandb config
            if "wandb" in config:
                del config["wandb"]

            # Save cluster config
            cluster_config_path = configs_dir / f"cluster_{config_file.name}"
            with open(cluster_config_path, "w") as f:
                yaml.safe_dump(config, f, default_flow_style=False, indent=2)

    def _update_object_detection_configs(self, cluster_config: Dict[str, Any]):
        """Update object detection configs for cluster."""
        configs_dir = Path("experiments/object_detection/configs")

        if not configs_dir.exists():
            return

        for config_file in configs_dir.glob("*.yaml"):
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)

            # Update data paths
            if self.quick:
                config["data"]["data_dir"] = str(
                    self.local_data_dir / "sample_COCO2017"
                )
            else:
                config["data"]["data_dir"] = str(self.local_data_dir / "COCO2017")

            # Update for cluster
            config.update(cluster_config)

            # Adjust batch size for multi-GPU
            original_batch_size = config["training"].get("batch_size", 8)
            config["training"]["batch_size"] = max(
                1, original_batch_size // self.num_gpus
            )
            config["training"]["effective_batch_size"] = original_batch_size

            # Adjust epochs for quick mode
            if self.quick:
                config["training"]["epochs"] = 2

            # Remove wandb config
            if "wandb" in config:
                del config["wandb"]

            # Save cluster config
            cluster_config_path = configs_dir / f"cluster_{config_file.name}"
            with open(cluster_config_path, "w") as f:
                yaml.safe_dump(config, f, default_flow_style=False, indent=2)

    def _update_ablation_configs(self, cluster_config: Dict[str, Any]):
        """Update ablation configs for cluster."""
        configs_dir = Path("experiments/ablations/configs")

        if not configs_dir.exists():
            return

        for config_file in configs_dir.glob("*.yaml"):
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)

            # Update for cluster
            config.update(cluster_config)

            # Use ISIC data for ablations
            if self.quick:
                config["data"]["data_dir"] = str(
                    self.local_data_dir / "sample_ISIC2018"
                )
            else:
                config["data"]["data_dir"] = str(self.local_data_dir / "ISIC2018")

            # Adjust for quick mode
            if self.quick:
                config["training"]["epochs"] = 2

            # Remove wandb config
            if "wandb" in config:
                del config["wandb"]

            # Save cluster config
            cluster_config_path = configs_dir / f"cluster_{config_file.name}"
            with open(cluster_config_path, "w") as f:
                yaml.safe_dump(config, f, default_flow_style=False, indent=2)

    def _create_distributed_training_configs(self):
        """Create configuration for distributed training."""
        # Create distributed training script config
        dist_config = {
            "world_size": self.num_gpus,
            "rank": 0,  # Will be set by launcher
            "local_rank": 0,  # Will be set by launcher
            "dist_backend": "nccl",
            "dist_url": "env://",  # Use environment variables
            "multiprocessing_distributed": True,
            "find_unused_parameters": True,
        }

        # Save distributed config
        dist_config_dir = Path("experiments/configs")
        dist_config_dir.mkdir(exist_ok=True)

        with open(dist_config_dir / "distributed.yaml", "w") as f:
            yaml.safe_dump(dist_config, f, default_flow_style=False, indent=2)

    def create_slurm_scripts(self):
        """Create SLURM job scripts for cluster submission."""
        print("Creating SLURM job scripts...")

        scripts_dir = Path("cluster_scripts")
        scripts_dir.mkdir(exist_ok=True)

        # Main experiment SLURM script
        slurm_script = f"""#!/bin/bash
#SBATCH --job-name=rat_experiments
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={min(32, self.num_gpus * 4)}
#SBATCH --gres=gpu:{self.num_gpus}
#SBATCH --time=24:00:00
#SBATCH --mem=0  # Use all available memory
#SBATCH --output={self.network_results_dir}/slurm_logs/rat_experiments_%j.out
#SBATCH --error={self.network_results_dir}/slurm_logs/rat_experiments_%j.err

# Setup environment
source ~/.bashrc
conda activate rat_env  # or your environment name

# Set environment variables for distributed training
export MASTER_ADDR=localhost
export MASTER_PORT=12355
export WORLD_SIZE={self.num_gpus}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # Adjust based on available GPUs

# Setup local data
echo "Setting up local data on node $SLURM_NODEID"
python scripts/setup_cluster.py \\
    --local_data_dir {self.local_data_dir} \\
    --network_results_dir {self.network_results_dir} \\
    --network_checkpoints_dir {self.network_checkpoints_dir} \\
    --num_gpus {self.num_gpus} \\
    --node_id $SLURM_NODEID

# Run experiments with distributed training
echo "Starting RAT experiments"
torchrun \\
    --nnodes=1 \\
    --nproc_per_node={self.num_gpus} \\
    --master_addr=localhost \\
    --master_port=12355 \\
    experiments/run_distributed_experiments.py \\
    --config_dir experiments/configs \\
    --results_dir {self.network_results_dir} \\
    --checkpoint_dir {self.network_checkpoints_dir}

echo "Experiments completed"

# Clean up local data if needed
# rm -rf {self.local_data_dir}/*
"""

        with open(scripts_dir / "run_experiments.slurm", "w") as f:
            f.write(slurm_script)

        # Quick test SLURM script
        quick_slurm_script = f"""#!/bin/bash
#SBATCH --job-name=rat_quick_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --output={self.network_results_dir}/slurm_logs/rat_quick_test_%j.out
#SBATCH --error={self.network_results_dir}/slurm_logs/rat_quick_test_%j.err

# Setup environment
source ~/.bashrc
conda activate rat_env

export MASTER_ADDR=localhost
export MASTER_PORT=12356
export WORLD_SIZE=2
export CUDA_VISIBLE_DEVICES=0,1

# Quick test setup
python scripts/setup_cluster.py \\
    --quick \\
    --local_data_dir {self.local_data_dir} \\
    --network_results_dir {self.network_results_dir} \\
    --network_checkpoints_dir {self.network_checkpoints_dir} \\
    --num_gpus 2

# Run quick experiments
torchrun \\
    --nnodes=1 \\
    --nproc_per_node=2 \\
    --master_addr=localhost \\
    --master_port=12356 \\
    experiments/run_distributed_experiments.py \\
    --config_dir experiments/configs \\
    --results_dir {self.network_results_dir} \\
    --checkpoint_dir {self.network_checkpoints_dir} \\
    --quick
"""

        with open(scripts_dir / "run_quick_test.slurm", "w") as f:
            f.write(quick_slurm_script)

        # Make scripts executable
        os.chmod(scripts_dir / "run_experiments.slurm", 0o755)
        os.chmod(scripts_dir / "run_quick_test.slurm", 0o755)

        print(f"SLURM scripts created in {scripts_dir}/")

    def create_monitoring_setup(self):
        """Create monitoring and logging setup."""
        print("Setting up monitoring and logging...")

        # Create TensorBoard launch script
        scripts_dir = Path("cluster_scripts")
        scripts_dir.mkdir(exist_ok=True)

        tensorboard_script = f"""#!/bin/bash
# TensorBoard monitoring script
# Run this on a login node or dedicated monitoring node

# Load environment
source ~/.bashrc
conda activate rat_env

# Start TensorBoard server
tensorboard \\
    --logdir {self.network_results_dir}/tensorboard_logs \\
    --port 6006 \\
    --host 0.0.0.0 \\
    --reload_interval 30

echo "TensorBoard running at http://$(hostname):6006"
"""

        with open(scripts_dir / "start_tensorboard.sh", "w") as f:
            f.write(tensorboard_script)

        os.chmod(scripts_dir / "start_tensorboard.sh", 0o755)

        # Create log directories
        log_dirs = [
            self.network_results_dir / "tensorboard_logs",
            self.network_results_dir / "slurm_logs",
            self.network_results_dir / "experiment_logs",
        ]

        for log_dir in log_dirs:
            log_dir.mkdir(parents=True, exist_ok=True)

    def create_cleanup_scripts(self):
        """Create cleanup scripts for managing local storage."""
        scripts_dir = Path("cluster_scripts")
        scripts_dir.mkdir(exist_ok=True)

        cleanup_script = f"""#!/bin/bash
# Cleanup script for local node storage
# Run this after experiments complete or when needed

echo "Cleaning up local data directory: {self.local_data_dir}"

# Remove datasets (keep directory structure)
rm -rf {self.local_data_dir}/*/

# Remove temporary files
find {self.local_data_dir} -name "*.tmp" -delete
find {self.local_data_dir} -name ".cache" -type d -exec rm -rf {{}} +

# Show remaining disk usage
echo "Remaining disk usage:"
du -sh {self.local_data_dir}

echo "Cleanup completed"
"""

        with open(scripts_dir / "cleanup_local.sh", "w") as f:
            f.write(cleanup_script)

        os.chmod(scripts_dir / "cleanup_local.sh", 0o755)

    def verify_cluster_setup(self):
        """Verify cluster setup is working correctly."""
        print("Verifying cluster setup...")

        # Check GPU availability
        try:
            import torch

            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                print(f"✓ CUDA available with {gpu_count} GPUs")

                for i in range(min(gpu_count, self.num_gpus)):
                    gpu_name = torch.cuda.get_device_name(i)
                    memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                    print(f"  GPU {i}: {gpu_name} ({memory:.1f}GB)")
            else:
                print("✗ CUDA not available")
                return False
        except ImportError:
            print("✗ PyTorch not available")
            return False

        # Check data directories
        data_checks = [
            (self.local_data_dir, "Local data directory"),
            (self.network_results_dir, "Network results directory"),
            (self.network_checkpoints_dir, "Network checkpoints directory"),
        ]

        for path, description in data_checks:
            if path.exists():
                print(f"✓ {description}: {path}")
            else:
                print(f"✗ {description} missing: {path}")
                return False

        print("✓ Cluster setup verification completed successfully")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Setup RAT experiments for cluster computing"
    )
    parser.add_argument(
        "--local_data_dir",
        type=str,
        default="/tmp/rat_data",
        help="Local directory for datasets (fast storage)",
    )
    parser.add_argument(
        "--network_results_dir",
        type=str,
        default="/shared/results/rat_experiments",
        help="Network directory for experiment results",
    )
    parser.add_argument(
        "--network_checkpoints_dir",
        type=str,
        default="/shared/checkpoints/rat",
        help="Network directory for model checkpoints",
    )
    parser.add_argument(
        "--num_gpus", type=int, default=8, help="Number of GPUs available on the node"
    )
    parser.add_argument(
        "--node_id", type=str, default="0", help="Node ID (usually from SLURM_NODEID)"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Quick setup with sample data for testing"
    )

    args = parser.parse_args()

    print("🚀 Setting up RAT experiments for cluster computing")
    print("=" * 60)
    print(f"Local data directory: {args.local_data_dir}")
    print(f"Network results directory: {args.network_results_dir}")
    print(f"Network checkpoints directory: {args.network_checkpoints_dir}")
    print(f"Number of GPUs: {args.num_gpus}")
    print(f"Node ID: {args.node_id}")
    print(f"Quick mode: {args.quick}")
    print("=" * 60)

    # Initialize cluster setup
    cluster_setup = ClusterSetup(
        args.local_data_dir,
        args.network_results_dir,
        args.network_checkpoints_dir,
        args.num_gpus,
        args.node_id,
        args.quick,
    )

    try:
        # Setup data locally
        cluster_setup.setup_data_locally()

        # Create cluster-optimized configs
        cluster_setup.create_cluster_configs()

        # Create SLURM scripts
        cluster_setup.create_slurm_scripts()

        # Setup monitoring
        cluster_setup.create_monitoring_setup()

        # Create cleanup scripts
        cluster_setup.create_cleanup_scripts()

        # Verify setup
        success = cluster_setup.verify_cluster_setup()

        if success:
            print("\n🎉 Cluster setup completed successfully!")
            print("\nNext steps:")
            print("1. Submit job: sbatch cluster_scripts/run_experiments.slurm")
            print("2. Monitor: cluster_scripts/start_tensorboard.sh")
            print("3. Quick test: sbatch cluster_scripts/run_quick_test.slurm")
        else:
            print("\n⚠️ Setup completed with warnings. Check verification messages.")

    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
