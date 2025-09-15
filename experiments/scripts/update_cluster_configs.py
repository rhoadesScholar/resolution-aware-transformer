#!/usr/bin/env python3
"""
Simple Cluster Configuration Update Script

Updates configuration files for cluster-optimized multi-GPU training.
Switches from WandB to TensorBoard logging.

Usage:
    python update_cluster_configs.py --data_dir /local/data --results_dir /shared/results --num_gpus 8
"""

import argparse
import yaml
from pathlib import Path


def update_config_for_cluster(config_path, updates):
    """Update a single config file for cluster training."""
    print(f"Updating {config_path.name} for cluster...")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Remove WandB configuration
    if "wandb" in config:
        del config["wandb"]

    # Add TensorBoard logging
    config["logging"] = {
        "backend": "tensorboard",
        "log_dir": updates["tensorboard_dir"],
        "log_freq": 50,
        "save_predictions": True,
    }

    # Add cluster configuration
    config["cluster"] = {
        "num_gpus": updates["num_gpus"],
        "distributed_backend": "nccl",
        "find_unused_parameters": True,
    }

    # Update data configuration
    if "data" not in config:
        config["data"] = {}

    config["data"].update(
        {
            "data_dir": updates["data_dir"],
            "num_workers": min(16, updates["num_gpus"] * 2),
            "pin_memory": True,
            "prefetch_factor": 2,
            "persistent_workers": True,
        }
    )

    # Update training configuration for multi-GPU
    if "training" not in config:
        config["training"] = {}

    # Adjust batch size for multi-GPU
    original_batch_size = config["training"].get("batch_size", 16)
    config["training"].update(
        {
            "batch_size": max(1, original_batch_size // updates["num_gpus"]),
            "effective_batch_size": original_batch_size,
            "mixed_precision": True,
            "gradient_accumulation_steps": max(1, 32 // updates["num_gpus"]),
            "save_freq": 5,
            "eval_freq": 2,
        }
    )

    # Update checkpoints configuration
    config["checkpoints"] = {
        "save_dir": updates["checkpoint_dir"],
        "save_top_k": 3,
        "save_last": True,
        "monitor": "val_dice_score",
        "mode": "max",
    }

    # Save cluster-optimized config
    cluster_config_path = config_path.parent / f"cluster_{config_path.name}"
    with open(cluster_config_path, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False, indent=2)

    print(f"  ✓ Saved cluster config: {cluster_config_path}")


def create_distributed_launcher():
    """Create a simple distributed training launcher script."""
    launcher_script = """#!/bin/bash
# Distributed Training Launcher for RAT Experiments

# Set environment variables
export MASTER_ADDR=localhost
export MASTER_PORT=12355
export WORLD_SIZE=${1:-8}  # Default to 8 GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Configuration
CONFIG_DIR=${2:-"experiments/medical_segmentation/configs"}
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
torchrun \\
    --nnodes=1 \\
    --nproc_per_node=$WORLD_SIZE \\
    --master_addr=localhost \\
    --master_port=12355 \\
    experiments/train_distributed.py \\
    --config_dir "$CONFIG_DIR" \\
    --results_dir "$RESULTS_DIR" \\
    --checkpoint_dir "$CHECKPOINT_DIR"

echo "Training completed"
"""

    script_path = Path("cluster_scripts/launch_distributed_training.sh")
    script_path.parent.mkdir(exist_ok=True)

    with open(script_path, "w") as f:
        f.write(launcher_script)

    script_path.chmod(0o755)
    print(f"Created distributed launcher: {script_path}")


def create_simple_trainer():
    """Create a simplified distributed trainer."""
    trainer_code = '''#!/usr/bin/env python3
"""
Simple Distributed Trainer for RAT Experiments
"""

import argparse
import os
import sys
import logging
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import yaml

# Try different tensorboard imports
try:
    from torch.utils.tensorboard.writer import SummaryWriter
except ImportError:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        try:
            from tensorboardX import SummaryWriter
        except ImportError:
            print("Warning: No TensorBoard found. Logging disabled.")
            SummaryWriter = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_distributed():
    """Initialize distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        
        logger.info(f"Distributed training: rank {rank}/{world_size}")
        return rank, world_size, local_rank
    else:
        return 0, 1, 0


def create_simple_model():
    """Create a simple model for testing."""
    import torch.nn as nn
    
    class SimpleSegmentationModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 1, 1),  # Binary segmentation
            )
        
        def forward(self, x):
            return self.encoder(x)
    
    return SimpleSegmentationModel()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    args = parser.parse_args()
    
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')
    
    # Setup logging (only on rank 0)
    writer = None
    if rank == 0 and SummaryWriter is not None:
        log_dir = Path(args.results_dir) / "tensorboard_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(log_dir))
    
    # Find config files
    config_dir = Path(args.config_dir)
    config_files = list(config_dir.glob("cluster_*.yaml"))
    if not config_files:
        config_files = list(config_dir.glob("*.yaml"))
    
    logger.info(f"Found {len(config_files)} config files")
    
    for config_file in config_files:
        if rank == 0:
            logger.info(f"Training with config: {config_file.name}")
        
        # Load config
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Create model
        model = create_simple_model().to(device)
        
        if world_size > 1:
            model = DDP(model, device_ids=[local_rank])
        
        # Simple training setup
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        criterion = torch.nn.BCEWithLogitsLoss()
        
        # Dummy training loop
        for epoch in range(config.get('training', {}).get('epochs', 3)):
            model.train()
            
            # Create dummy batch
            batch_size = config.get('training', {}).get('batch_size', 4)
            dummy_images = torch.randn(batch_size, 3, 256, 256).to(device)
            dummy_masks = torch.randint(0, 2, (batch_size, 1, 256, 256)).float().to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(dummy_images)
            loss = criterion(outputs, dummy_masks)
            loss.backward()
            optimizer.step()
            
            if rank == 0:
                logger.info(f"Epoch {epoch}, Loss: {loss.item():.6f}")
                if writer is not None:
                    writer.add_scalar('Train/Loss', loss.item(), epoch)
        
        if rank == 0:
            logger.info(f"Completed training with {config_file.name}")
    
    # Cleanup
    if writer is not None:
        writer.close()
    
    if world_size > 1:
        dist.destroy_process_group()
    
    logger.info("All experiments completed")


if __name__ == "__main__":
    main()
'''

    trainer_path = Path("experiments/train_distributed.py")
    with open(trainer_path, "w") as f:
        f.write(trainer_code)

    trainer_path.chmod(0o755)
    print(f"Created simple distributed trainer: {trainer_path}")


def main():
    parser = argparse.ArgumentParser(description="Update configs for cluster training")
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Local data directory"
    )
    parser.add_argument(
        "--results_dir", type=str, required=True, help="Network results directory"
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, required=True, help="Network checkpoint directory"
    )
    parser.add_argument("--num_gpus", type=int, default=8, help="Number of GPUs")

    args = parser.parse_args()

    # Setup paths
    tensorboard_dir = f"{args.results_dir}/tensorboard_logs"

    updates = {
        "data_dir": args.data_dir,
        "tensorboard_dir": tensorboard_dir,
        "checkpoint_dir": args.checkpoint_dir,
        "num_gpus": args.num_gpus,
    }

    print("Updating experiment configurations for cluster training...")

    # Update medical segmentation configs
    med_configs_dir = Path("experiments/medical_segmentation/configs")
    if med_configs_dir.exists():
        for config_file in med_configs_dir.glob("*.yaml"):
            if not config_file.name.startswith("cluster_"):
                update_config_for_cluster(config_file, updates)

    # Update object detection configs
    obj_configs_dir = Path("experiments/object_detection/configs")
    if obj_configs_dir.exists():
        for config_file in obj_configs_dir.glob("*.yaml"):
            if not config_file.name.startswith("cluster_"):
                update_config_for_cluster(config_file, updates)

    # Update ablation configs
    abl_configs_dir = Path("experiments/ablations/configs")
    if abl_configs_dir.exists():
        for config_file in abl_configs_dir.glob("*.yaml"):
            if not config_file.name.startswith("cluster_"):
                update_config_for_cluster(config_file, updates)

    # Create cluster scripts
    create_distributed_launcher()
    create_simple_trainer()

    print("\n🎉 Cluster configuration completed!")
    print("\nNext steps:")
    print("1. Test setup: bash cluster_scripts/launch_distributed_training.sh 2")
    print("2. Full training: bash cluster_scripts/launch_distributed_training.sh 8")
    print("3. Monitor training: tensorboard --logdir", tensorboard_dir)


if __name__ == "__main__":
    main()
