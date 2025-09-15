#!/usr/bin/env python3
"""
Distributed Experiment Runner for Resolution Aware Transformer

This script handles distributed training across multiple GPUs on a single node.
It replaces WandB with TensorBoard and optimizes for cluster computing.

Usage:
    # Run with torchrun for distributed training
    torchrun --nnodes=1 --nproc_per_node=8 run_distributed_experiments.py --config_dir configs

    # Single GPU test
    python run_distributed_experiments.py --config_dir configs --no_distributed
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from torch.utils.tensorboard.writer import SummaryWriter
except ImportError:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        from tensorboardX import SummaryWriter
import yaml


# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DistributedExperimentRunner:
    """Handles distributed training of RAT experiments."""

    def __init__(self, config: Dict[str, Any], rank: int = 0, world_size: int = 1):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.local_rank = rank % torch.cuda.device_count()

        # Setup device
        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.device)

        # Initialize tensorboard writer (only on rank 0)
        self.writer = None
        if self.rank == 0:
            log_dir = Path(config["logging"]["log_dir"]) / config["experiment_name"]
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(log_dir))
            logger.info(f"TensorBoard logging to: {log_dir}")

        # Setup directories
        self.results_dir = Path(config.get("results_dir", "./results"))
        self.checkpoint_dir = Path(config.get("checkpoint_dir", "./checkpoints"))
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def setup_distributed(self):
        """Initialize distributed training."""
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            self.rank = int(os.environ["RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.local_rank = int(os.environ["LOCAL_RANK"])

        if self.world_size > 1:
            dist.init_process_group(
                backend=self.config["cluster"]["distributed_backend"],
                init_method="env://",
                world_size=self.world_size,
                rank=self.rank,
            )
            logger.info(f"Distributed training: rank {self.rank}/{self.world_size}")

    def cleanup_distributed(self):
        """Clean up distributed training."""
        if self.world_size > 1:
            dist.destroy_process_group()

    def create_model(self, model_config: Dict[str, Any]):
        """Create and setup model for distributed training."""
        # Simplified model creation - import actual RAT model
        try:
            from resolution_aware_transformer import ResolutionAwareTransformer

            model = ResolutionAwareTransformer()
        except ImportError:
            # Fallback to a simple model for testing
            import torch.nn as nn

            class SimpleTestModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.encoder = nn.Sequential(
                        nn.Conv2d(3, 64, 3, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(64, 128, 3, padding=1),
                        nn.ReLU(),
                        nn.AdaptiveAvgPool2d((1, 1)),
                        nn.Flatten(),
                        nn.Linear(128, 1),
                    )

                def forward(self, x):
                    return self.encoder(x).unsqueeze(-1).unsqueeze(-1)

            model = SimpleTestModel()

        model = model.to(self.device)

        # Mixed precision will be handled in the training loop if configured

        # Enable model compilation if configured
        if self.config["training"].get("compile_model", False):
            try:
                model = torch.compile(model)
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}")

        # Wrap with DDP for distributed training
        if self.world_size > 1:
            model = DDP(
                model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=self.config["cluster"].get(
                    "find_unused_parameters", True
                ),
            )

        return model

    def _create_baseline_model(self, model_config: Dict[str, Any]):
        """Create baseline models for comparison."""
        # This would import and create baseline models like U-Net, etc.
        # For now, return a placeholder
        import torch.nn as nn

        class PlaceholderModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 1, 3, padding=1)

            def forward(self, x):
                return self.conv(x)

        return PlaceholderModel()

    def create_dataloader(self, data_config: Dict[str, Any], split: str = "train"):
        """Create distributed dataloader."""
        # Import dataset based on experiment type
        dataset_name = data_config.get("dataset", "isic")

        if dataset_name == "isic":
            from experiments.common.datasets import ISICDataset

            dataset = ISICDataset(
                data_dir=data_config["data_dir"],
                split=split,
                image_size=data_config.get("image_size", 256),
                multi_scale=data_config.get("multi_scale", False),
            )
        elif dataset_name == "coco":
            from experiments.common.datasets import COCODataset

            dataset = COCODataset(
                data_dir=data_config["data_dir"],
                split=split,
                image_size=data_config.get("image_size", 512),
                multi_scale=data_config.get("multi_scale", False),
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        # Create distributed sampler
        sampler = None
        if self.world_size > 1:
            sampler = torch.utils.data.DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=(split == "train"),
            )

        # Create dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=data_config.get("batch_size", 8),
            shuffle=(split == "train" and sampler is None),
            sampler=sampler,
            num_workers=data_config.get("num_workers", 4),
            pin_memory=data_config.get("pin_memory", True),
            persistent_workers=data_config.get("persistent_workers", True),
            prefetch_factor=data_config.get("prefetch_factor", 2),
        )

        return dataloader, sampler

    def train_epoch(self, model, dataloader, optimizer, criterion, scaler, epoch):
        """Train for one epoch."""
        model.train()
        total_loss = 0.0
        num_batches = len(dataloader)

        for batch_idx, batch in enumerate(dataloader):
            # Move data to device
            images = batch[0].to(self.device, non_blocking=True)
            targets = batch[1].to(self.device, non_blocking=True)

            # Forward pass with mixed precision
            optimizer.zero_grad()

            if self.config["training"].get("mixed_precision", False):
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, targets)

                # Backward pass with gradient scaling
                scaler.scale(loss).backward()

                # Gradient clipping
                if self.config["training"].get("grad_clip", 0) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.config["training"]["grad_clip"]
                    )

                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, targets)
                loss.backward()

                # Gradient clipping
                if self.config["training"].get("grad_clip", 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.config["training"]["grad_clip"]
                    )

                optimizer.step()

            total_loss += loss.item()

            # Log training progress
            if batch_idx % self.config["logging"]["log_freq"] == 0 and self.rank == 0:
                lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Epoch {epoch}, Batch {batch_idx}/{num_batches}, "
                    f"Loss: {loss.item():.6f}, LR: {lr:.6f}"
                )

                # TensorBoard logging
                if self.writer is not None:
                    global_step = epoch * num_batches + batch_idx
                    self.writer.add_scalar("Train/Loss", loss.item(), global_step)
                    self.writer.add_scalar("Train/LearningRate", lr, global_step)

        avg_loss = total_loss / num_batches

        # Synchronize loss across processes
        if self.world_size > 1:
            loss_tensor = torch.tensor(avg_loss, device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = loss_tensor.item() / self.world_size

        return avg_loss

    def validate(self, model, dataloader, criterion, epoch):
        """Validate model performance."""
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                images = batch[0].to(self.device, non_blocking=True)
                targets = batch[1].to(self.device, non_blocking=True)

                if self.config["training"].get("mixed_precision", False):
                    with torch.cuda.amp.autocast():
                        outputs = model(images)
                        loss = criterion(outputs, targets)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, targets)

                total_loss += loss.item()

                # Calculate accuracy/metrics (simplified)
                if outputs.shape[1] == 1:  # Binary segmentation
                    pred = (torch.sigmoid(outputs) > 0.5).float()
                    correct = (pred == targets).float().sum()
                    total_correct += correct.item()
                    total_samples += targets.numel()

        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        # Synchronize metrics across processes
        if self.world_size > 1:
            metrics = torch.tensor([avg_loss, accuracy], device=self.device)
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            avg_loss, accuracy = (metrics / self.world_size).tolist()

        # Log validation results
        if self.rank == 0:
            logger.info(
                f"Validation - Epoch {epoch}, Loss: {avg_loss:.6f}, Accuracy: {accuracy:.4f}"
            )
            if self.writer is not None:
                self.writer.add_scalar("Val/Loss", avg_loss, epoch)
                self.writer.add_scalar("Val/Accuracy", accuracy, epoch)

        return avg_loss, accuracy

    def save_checkpoint(
        self, model, optimizer, scheduler, epoch, metrics, is_best=False
    ):
        """Save model checkpoint."""
        if self.rank != 0:  # Only save on rank 0
            return

        # Get model state dict (unwrap DDP if needed)
        model_state = (
            model.module.state_dict()
            if hasattr(model, "module")
            else model.state_dict()
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "metrics": metrics,
            "config": self.config,
        }

        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / "latest_checkpoint.pth"
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_checkpoint.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best checkpoint: {best_path}")

        # Save periodic checkpoints
        if epoch % self.config["training"]["save_freq"] == 0:
            epoch_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
            torch.save(checkpoint, epoch_path)

    def run_experiment(self, config_path: str):
        """Run a single experiment."""
        logger.info(f"Starting experiment: {config_path}")

        # Load configuration
        with open(config_path, "r") as f:
            exp_config = yaml.safe_load(f)

        # Merge with base config
        self.config.update(exp_config)

        # Create model
        model = self.create_model(self.config["model"])

        # Create data loaders
        train_loader, train_sampler = self.create_dataloader(
            self.config["data"], "train"
        )
        val_loader, _ = self.create_dataloader(self.config["data"], "val")

        # Setup training components
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config["training"]["learning_rate"],
            weight_decay=self.config["training"]["weight_decay"],
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config["training"]["epochs"]
        )

        criterion = torch.nn.BCEWithLogitsLoss()  # For binary segmentation
        scaler = (
            torch.cuda.amp.GradScaler()
            if self.config["training"].get("mixed_precision", False)
            else None
        )

        # Training loop
        best_metric = float("-inf")

        for epoch in range(self.config["training"]["epochs"]):
            # Set epoch for distributed sampler
            if train_sampler:
                train_sampler.set_epoch(epoch)

            # Train
            train_loss = self.train_epoch(
                model, train_loader, optimizer, criterion, scaler, epoch
            )

            # Validate
            if epoch % self.config["training"]["eval_freq"] == 0:
                val_loss, val_accuracy = self.validate(
                    model, val_loader, criterion, epoch
                )

                # Check if best model
                is_best = val_accuracy > best_metric
                if is_best:
                    best_metric = val_accuracy

                # Save checkpoint
                metrics = {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                }
                self.save_checkpoint(
                    model, optimizer, scheduler, epoch, metrics, is_best
                )

            # Update learning rate
            if scheduler:
                scheduler.step()

        logger.info(f"Experiment completed: {config_path}")


def main():
    parser = argparse.ArgumentParser(description="Run distributed RAT experiments")
    parser.add_argument(
        "--config_dir", type=str, required=True, help="Configuration directory"
    )
    parser.add_argument(
        "--results_dir", type=str, default="./results", help="Results directory"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="Checkpoint directory",
    )
    parser.add_argument("--quick", action="store_true", help="Quick test mode")
    parser.add_argument(
        "--no_distributed", action="store_true", help="Disable distributed training"
    )

    args = parser.parse_args()

    # Setup base configuration
    base_config = {
        "results_dir": args.results_dir,
        "checkpoint_dir": args.checkpoint_dir,
        "logging": {
            "backend": "tensorboard",
            "log_dir": f"{args.results_dir}/tensorboard_logs",
            "log_freq": 50,
        },
        "cluster": {"distributed_backend": "nccl", "find_unused_parameters": True},
        "training": {
            "mixed_precision": True,
            "compile_model": True,
            "grad_clip": 1.0,
            "save_freq": 5,
            "eval_freq": 2,
        },
    }

    # Initialize distributed training
    if not args.no_distributed and torch.cuda.device_count() > 1:
        # Get rank and world size from environment
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        runner = DistributedExperimentRunner(base_config, rank, world_size)
        runner.setup_distributed()
    else:
        runner = DistributedExperimentRunner(base_config, 0, 1)

    try:
        # Find and run experiments
        config_dir = Path(args.config_dir)
        config_files = list(config_dir.glob("cluster_*.yaml"))

        if not config_files:
            logger.warning(f"No cluster config files found in {config_dir}")
            config_files = list(config_dir.glob("*.yaml"))

        if args.quick:
            # Run only quick test configs
            config_files = [f for f in config_files if "quick" in f.name][:2]

        logger.info(f"Found {len(config_files)} configuration files")

        for config_file in config_files:
            runner.run_experiment(str(config_file))

        logger.info("All experiments completed successfully")

    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise
    finally:
        # Cleanup
        if runner.writer:
            runner.writer.close()
        runner.cleanup_distributed()


if __name__ == "__main__":
    main()
