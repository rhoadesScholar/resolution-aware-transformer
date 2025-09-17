"""Training script for medical segmentation experiment."""

import os

# Set OMP_NUM_THREADS to 1 to avoid thread oversubscription in distributed training
os.environ.setdefault("OMP_NUM_THREADS", "1")

# CUDA memory and debugging environment variables
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

import argparse
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import yaml

# DeepSpeed for memory optimization and large model training
try:
    import deepspeed

    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False
    print("Warning: DeepSpeed not available. Install with: pip install deepspeed")

# TensorBoard for logging
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    try:
        from tensorboardX import SummaryWriter
    except ImportError:
        SummaryWriter = None

# Add common utilities to path
sys.path.append(str(Path(__file__).parent.parent / "common"))
from datasets import ISICDataset
from metrics import SegmentationEvaluator
from models import create_model, get_model_config, save_model
from utils import (
    ExperimentTracker,
    count_parameters,
    get_device,
    set_seed,
    setup_logging,
    adjust_config_for_gpu_memory,
)


def setup_distributed():
    """Initialize distributed training."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

        return rank, world_size, local_rank
    else:
        return 0, 1, 0


def get_device_for_distributed(local_rank=None):
    """Get appropriate device for distributed training."""
    if local_rank is not None:
        return torch.device(f"cuda:{local_rank}")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def create_deepspeed_config(args, config):
    """Create DeepSpeed configuration dictionary."""
    if not DEEPSPEED_AVAILABLE or not args.deepspeed:
        return None

    # Use provided config file or create default
    if args.deepspeed_config:
        import json

        with open(args.deepspeed_config, "r") as f:
            ds_config = json.load(f)
        return ds_config

    # Create default DeepSpeed config for medical segmentation
    ds_config = {
        "train_batch_size": config["training"]["batch_size"] * 8,  # Assumes 8 GPUs
        "train_micro_batch_size_per_gpu": config["training"]["batch_size"],
        "gradient_accumulation_steps": 1,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": config["training"]["learning_rate"],
                "weight_decay": config["training"].get("weight_decay", 1e-4),
                "betas": [0.9, 0.999],
                "eps": 1e-8,
            },
        },
        "scheduler": (
            {
                "type": "WarmupCosineLR",
                "params": {
                    "total_num_steps": config["training"]["epochs"] * 100,  # Estimate
                    "warmup_num_steps": 100,
                },
            }
            if config["training"].get("scheduler") == "cosine"
            else None
        ),
        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1,
        },
        "gradient_clipping": config.get("grad_clip", 1.0),
        "zero_optimization": (
            {
                "stage": args.zero_stage,
                "allgather_partitions": True,
                "allgather_bucket_size": 2e8,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 2e8,
                "contiguous_gradients": True,
                "cpu_offload": args.zero_stage == 3,  # Enable CPU offload for stage 3
            }
            if args.zero_stage > 0
            else None
        ),
        "activation_checkpointing": {
            "partition_activations": True,
            "cpu_checkpointing": True,
            "contiguous_memory_optimization": False,
            "number_checkpoints": 4,
            "synchronize_checkpoint_boundary": False,
            "profile": False,
        },
        "wall_clock_breakdown": False,
        "dump_state": False,
    }

    # Remove None values
    ds_config = {k: v for k, v in ds_config.items() if v is not None}
    if ds_config.get("zero_optimization") is not None:
        ds_config["zero_optimization"] = {
            k: v for k, v in ds_config["zero_optimization"].items() if v is not None
        }

    return ds_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train segmentation model on ISIC 2018"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--data_dir", type=str, help="Path to ISIC dataset")
    parser.add_argument(
        "--output_dir", type=str, default="./outputs", help="Output directory"
    )
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument(
        "--debug", action="store_true", help="Debug mode (smaller dataset)"
    )
    # DeepSpeed arguments
    parser.add_argument(
        "--deepspeed", action="store_true", help="Enable DeepSpeed optimization"
    )
    parser.add_argument(
        "--deepspeed_config", type=str, help="Path to DeepSpeed config file"
    )
    parser.add_argument(
        "--zero_stage",
        type=int,
        default=2,
        choices=[0, 1, 2, 3],
        help="DeepSpeed ZeRO stage (0=disabled, 1=optimizer, 2=optimizer+gradients, 3=all)",
    )

    # Parse args, but let DeepSpeed handle distributed args if present
    if DEEPSPEED_AVAILABLE:
        args = deepspeed.add_config_arguments(parser)
        args = parser.parse_args()
    else:
        args = parser.parse_args()

    return args


def load_config(config_path: str) -> dict:
    """Load and optimize experiment configuration."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Apply dynamic memory optimization
    config = adjust_config_for_gpu_memory(config)

    return config


def create_criterion(config: dict) -> nn.Module:
    """Create loss function."""
    loss_type = config.get("loss", "bce")

    if loss_type == "bce":
        return nn.BCEWithLogitsLoss()
    elif loss_type == "dice":
        return DiceLoss()
    elif loss_type == "combined":
        return CombinedLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


class DiceLoss(nn.Module):
    """Dice loss for segmentation."""

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)

        # Flatten
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    """Combined BCE + Dice loss."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = self.bce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return self.bce_weight * bce + self.dice_weight * dice


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    config: dict,
    args=None,
) -> dict:
    """Train for one epoch."""
    model.train()

    # Clear GPU memory before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    epoch_loss = 0.0
    evaluator = SegmentationEvaluator()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")

    for batch_idx, batch in enumerate(pbar):
        if config["model"]["multi_scale"]:
            # Multi-scale training
            images = [img.to(device) for img in batch["images"]]
            masks = [mask.to(device) for mask in batch["masks"]]

            # Forward pass
            outputs = model(images)

            # Calculate loss for each scale
            total_loss = 0
            for output, mask in zip(outputs, masks):
                loss = criterion(output, mask)
                total_loss += loss

            total_loss /= len(outputs)  # Average across scales

            # Use highest resolution for metrics
            with torch.no_grad():
                pred = torch.sigmoid(outputs[0])
                evaluator.update(pred, masks[0])

        else:
            # Single-scale training
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            # Forward pass
            outputs = model(images)
            total_loss = criterion(outputs, masks)

            # Update metrics
            with torch.no_grad():
                pred = torch.sigmoid(outputs)
                evaluator.update(pred, masks)

        # Backward pass
        if DEEPSPEED_AVAILABLE and args and args.deepspeed:
            # DeepSpeed handles backward pass and optimization
            model.backward(total_loss)
            model.step()
        else:
            # Traditional training
            optimizer.zero_grad()
            total_loss.backward()

            # Gradient clipping
            if config.get("grad_clip", 0) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])

            optimizer.step()

        epoch_loss += total_loss.item()

        # Update progress bar
        pbar.set_postfix(
            {
                "Loss": f"{total_loss.item():.4f}",
                "Avg Loss": f"{epoch_loss / (batch_idx + 1):.4f}",
            }
        )

        if config.get("debug", False) and batch_idx >= 5:
            break

    # Calculate epoch metrics
    metrics = evaluator.compute()
    metrics["loss"] = epoch_loss / len(dataloader)

    return metrics


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    config: dict,
) -> dict:
    """Validate for one epoch."""
    model.eval()

    epoch_loss = 0.0
    evaluator = SegmentationEvaluator()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")

    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            if config["model"]["multi_scale"]:
                # Multi-scale validation
                images = [img.to(device) for img in batch["images"]]
                masks = [mask.to(device) for mask in batch["masks"]]

                # Forward pass
                outputs = model(images)

                # Calculate loss for each scale
                total_loss = 0
                for output, mask in zip(outputs, masks):
                    loss = criterion(output, mask)
                    total_loss += loss

                total_loss /= len(outputs)

                # Use highest resolution for metrics
                pred = torch.sigmoid(outputs[0])
                evaluator.update(pred, masks[0])

            else:
                # Single-scale validation
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)

                # Forward pass
                outputs = model(images)
                total_loss = criterion(outputs, masks)

                # Update metrics
                pred = torch.sigmoid(outputs)
                evaluator.update(pred, masks)

            epoch_loss += total_loss.item()

            # Update progress bar
            pbar.set_postfix(
                {
                    "Loss": f"{total_loss.item():.4f}",
                    "Avg Loss": f"{epoch_loss / (batch_idx + 1):.4f}",
                }
            )

            if config.get("debug", False) and batch_idx >= 5:
                break

    # Calculate epoch metrics
    metrics = evaluator.compute()
    metrics["loss"] = epoch_loss / len(dataloader)

    return metrics


def main():
    args = parse_args()

    # Setup distributed training (DeepSpeed handles this if enabled)
    if DEEPSPEED_AVAILABLE and args.deepspeed:
        # DeepSpeed will handle distributed initialization
        deepspeed.init_distributed()
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank, world_size, local_rank = setup_distributed()

    # Load configuration
    config = load_config(args.config)

    # Create DeepSpeed config if enabled
    ds_config = create_deepspeed_config(args, config)

    # Set up experiment
    set_seed(config.get("seed", 42))
    device = get_device_for_distributed(local_rank)

    # Create output directory (only on rank 0)
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Wait for rank 0 to create directory
    if world_size > 1:
        dist.barrier()

    # Setup logging (only on rank 0)
    logger = None
    tracker = None
    if rank == 0:
        logger = setup_logging(str(output_dir), "medical_segmentation_train")
        tracker = ExperimentTracker("medical_segmentation", str(output_dir))
        tracker.log_config(config)
        if ds_config:
            tracker.log_config({"deepspeed_config": ds_config})
        tracker.start_timer()
    else:
        # Create a dummy logger for non-rank 0 processes
        import logging

        logger = logging.getLogger("medical_segmentation_train")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)

    # Initialize tensorboard if configured (only on rank 0)
    writer = None
    if rank == 0 and config.get("logging", {}).get("backend") == "tensorboard":
        if SummaryWriter is not None:
            log_dir = Path(config["logging"]["log_dir"]) / "medical_segmentation"
            log_dir.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(log_dir=str(log_dir))
            logger.info(f"TensorBoard logging to: {log_dir}")
        else:
            logger.warning("TensorBoard not available, skipping logging")

    logger.info("Starting medical segmentation experiment")
    logger.info(f"Device: {device}")
    logger.info(
        f"DeepSpeed enabled: {args.deepspeed if DEEPSPEED_AVAILABLE else False}"
    )
    if ds_config:
        logger.info(f"DeepSpeed ZeRO stage: {args.zero_stage}")
    logger.info(f"Config: {config}")

    # Create datasets
    data_dir = args.data_dir or config["data"]["data_dir"]

    train_dataset = ISICDataset(
        data_dir=data_dir,
        split="train",
        image_size=config["data"]["image_size"],
        multi_scale=config["model"]["multi_scale"],
        scales=config["model"].get("scales", [256, 128, 64]),
    )

    val_dataset = ISICDataset(
        data_dir=data_dir,
        split="val",
        image_size=config["data"]["image_size"],
        multi_scale=config["model"]["multi_scale"],
        scales=config["model"].get("scales", [256, 128, 64]),
    )

    # Create dataloaders with distributed support
    train_sampler = None
    val_sampler = None
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset, rank=rank, num_replicas=world_size
        )
        val_sampler = DistributedSampler(
            val_dataset, rank=rank, num_replicas=world_size, shuffle=False
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=(train_sampler is None),  # Don't shuffle if using DistributedSampler
        sampler=train_sampler,
        num_workers=config["data"].get("num_workers", 4),
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"] * 2,
        shuffle=False,
        sampler=val_sampler,
        num_workers=config["data"].get("num_workers", 4),
        pin_memory=True,
    )

    logger.info(f"Train dataset: {len(train_dataset)} samples")
    logger.info(f"Val dataset: {len(val_dataset)} samples")

    # Create model
    model_config = config["model"].copy()
    model_name = model_config.pop("name")

    if "config_name" in model_config:
        preset_config = get_model_config(model_config.pop("config_name"))
        model_config.update(preset_config)

    model = create_model(
        model_name=model_name, task="segmentation", num_classes=1, **model_config
    )

    logger.info(f"Model: {model}")
    logger.info(f"Parameters: {count_parameters(model):,}")

    # Initialize DeepSpeed or traditional training
    if DEEPSPEED_AVAILABLE and args.deepspeed:
        # DeepSpeed initialization
        model_engine, optimizer, _, _ = deepspeed.initialize(
            args=args, model=model, config=ds_config
        )

        # DeepSpeed handles device placement
        model = model_engine
        device = model_engine.device
        logger.info(f"DeepSpeed engine initialized on device: {device}")
        logger.info(f"DeepSpeed ZeRO stage: {args.zero_stage}")

        # DeepSpeed handles scheduler internally if configured
        scheduler = None

    else:
        # Traditional training setup
        # Clear CUDA cache before moving model to device
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Handle CUDA device assignment with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                model = model.to(device)
                logger.info(f"Successfully moved model to device: {device}")
                break
            except RuntimeError as e:
                if "CUDA" in str(e) and attempt < max_retries - 1:
                    logger.warning(
                        f"CUDA error on attempt {attempt + 1}/{max_retries}: {e}"
                    )
                    logger.info("Clearing CUDA cache and retrying...")
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    import time

                    time.sleep(2)  # Brief delay before retry
                else:
                    logger.error(
                        f"Failed to move model to device after {max_retries} attempts: {e}"
                    )
                    raise

        # Wrap model with DDP for distributed training
        if world_size > 1:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)
            logger.info("Model wrapped with DistributedDataParallel")

        # Create optimizer and scheduler for traditional training
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"].get("weight_decay", 1e-4),
        )

        if config["training"].get("scheduler", "cosine") == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config["training"]["epochs"]
            )
        else:
            scheduler = None

    # Create loss function
    criterion = create_criterion(config["training"])

    # Resume from checkpoint if specified
    start_epoch = 0
    best_dice = 0.0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_dice = checkpoint.get("best_dice", 0.0)
        logger.info(f"Resumed from epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, config["training"]["epochs"]):
        if rank == 0:
            logger.info(f"Epoch {epoch + 1}/{config['training']['epochs']}")

        # Set epoch for distributed sampler
        if world_size > 1 and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch + 1, config, args
        )
        if rank == 0:
            logger.info(
                f"Train - Loss: {train_metrics['loss']:.4f}, "
                f"Dice: {train_metrics['dice']:.4f}"
            )

        # Validate
        val_metrics = validate_epoch(
            model, val_loader, criterion, device, epoch + 1, config
        )
        if rank == 0:
            logger.info(
                f"Val - Loss: {val_metrics['loss']:.4f}, Dice: {val_metrics['dice']:.4f}"
            )

        # Update scheduler
        if scheduler:
            scheduler.step()

        # Log metrics (only on rank 0)
        if rank == 0 and tracker is not None:
            for key, value in train_metrics.items():
                tracker.log_metric(f"train_{key}", value, epoch)

            for key, value in val_metrics.items():
                tracker.log_metric(f"val_{key}", value, epoch)

        # TensorBoard logging
        if writer is not None:
            for key, value in train_metrics.items():
                writer.add_scalar(f"train/{key}", value, epoch)
            for key, value in val_metrics.items():
                writer.add_scalar(f"val/{key}", value, epoch)

        # Save best model
        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            save_model(
                model,
                str(output_dir / "best_model.pth"),
                epoch,
                optimizer,
                {"train_metrics": train_metrics, "val_metrics": val_metrics},
            )
            logger.info(f"New best model saved with Dice: {best_dice:.4f}")

        # Save checkpoint
        if (epoch + 1) % config["training"].get("save_freq", 10) == 0:
            save_model(
                model,
                str(output_dir / f"checkpoint_epoch_{epoch + 1}.pth"),
                epoch,
                optimizer,
                {"train_metrics": train_metrics, "val_metrics": val_metrics},
            )

    # Save final results (only on rank 0)
    if rank == 0 and tracker is not None:
        tracker.log_metric("final_best_dice", best_dice)
        duration = tracker.end_timer()
        logger.info(f"Training completed in {duration:.2f} seconds")
        logger.info(f"Best validation Dice: {best_dice:.4f}")
        tracker.save_results()

    # Close TensorBoard writer
    if writer is not None:
        writer.close()

    # Cleanup distributed training
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
