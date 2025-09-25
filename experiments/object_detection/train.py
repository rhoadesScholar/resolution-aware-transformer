"""Training script for object detection experiments."""

import os

# Set OMP_NUM_THREADS to 1 to avoid thread oversubscription in distributed training
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import yaml
import logging

# Initialize module-level logger
logger = logging.getLogger(__name__)

# DeepSpeed for memory optimization and large model training
try:
    import deepspeed

    DEEPSPEED_AVAILABLE = True
except ImportError:
    deepspeed = None
    DEEPSPEED_AVAILABLE = False
    logger.warning("DeepSpeed not available. Install with: pip install deepspeed")

# Add common utilities to path
sys.path.append(str(Path(__file__).parent.parent / "common"))
from datasets import COCODataset
from models import create_rat_detection_model
from utils import (
    AverageMeter,
    ExperimentTracker,
    get_device,
    set_seed,
    adjust_config_for_gpu_memory,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train object detection model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--data_dir", type=str, help="Path to COCO dataset")
    parser.add_argument(
        "--output_dir", type=str, default="./checkpoints", help="Output directory"
    )
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument(
        "--debug", action="store_true", help="Debug mode with small dataset"
    )

    # Distributed training arguments
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="Local rank for distributed training"
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=-1,
        help="Number of processes for distributed training",
    )
    parser.add_argument(
        "--distributed", action="store_true", help="Enable distributed training"
    )

    # DeepSpeed arguments for memory optimization and large model training
    parser.add_argument(
        "--deepspeed",
        action="store_true",
        help="Enable DeepSpeed for memory optimization and large model training",
    )
    parser.add_argument(
        "--zero_stage",
        type=int,
        default=2,
        choices=[1, 2, 3],
        help="DeepSpeed ZeRO optimization stage (1, 2, or 3)",
    )

    return parser.parse_args()


def create_deepspeed_config(zero_stage=2, train_micro_batch_size=1):
    """Create DeepSpeed configuration for memory optimization."""
    if zero_stage == 3:
        # Stage 3: Partition parameters, gradients, and optimizer states
        # Includes CPU offloading for maximum memory efficiency
        return {
            "train_micro_batch_size_per_gpu": train_micro_batch_size,
            "gradient_accumulation_steps": 1,
            "optimizer": {
                "type": "AdamW",
                "params": {"lr": 1e-4, "weight_decay": 0.01, "betas": [0.9, 0.999]},
            },
            "scheduler": {
                "type": "WarmupDecayLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": 1e-4,
                    "warmup_num_steps": 1000,
                    "total_num_steps": 50000,
                },
            },
            "fp16": {
                "enabled": True,
                "auto_cast": False,
                "loss_scale": 0,
                "initial_scale_power": 16,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "min_loss_scale": 1,
            },
            "zero_optimization": {
                "stage": 3,
                "cpu_offload": True,
                "contiguous_gradients": True,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 5e8,
                "allgather_bucket_size": 5e8,
                "stage3_prefetch_bucket_size": 5e8,
                "stage3_param_persistence_threshold": 1e6,
                "stage3_max_live_parameters": 1e9,
                "stage3_max_reuse_distance": 1e9,
                "gather_16bit_weights_on_model_save": True,
            },
            "activation_checkpointing": {
                "partition_activations": True,
                "cpu_checkpointing": True,
                "contiguous_memory_optimization": False,
                "number_checkpoints": 4,
                "synchronize_checkpoint_boundary": False,
                "profile": False,
            },
            "wall_clock_breakdown": False,
            "zero_allow_untested_optimizer": True,
        }
    else:
        # Stage 2: Partition gradients and optimizer states (default)
        return {
            "train_micro_batch_size_per_gpu": train_micro_batch_size,
            "gradient_accumulation_steps": 1,
            "optimizer": {
                "type": "AdamW",
                "params": {"lr": 1e-4, "weight_decay": 0.01, "betas": [0.9, 0.999]},
            },
            "scheduler": {
                "type": "WarmupDecayLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": 1e-4,
                    "warmup_num_steps": 1000,
                    "total_num_steps": 50000,
                },
            },
            "fp16": {
                "enabled": True,
                "auto_cast": False,
                "loss_scale": 0,
                "initial_scale_power": 16,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "min_loss_scale": 1,
            },
            "zero_optimization": {
                "stage": 2,
                "contiguous_gradients": True,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 5e8,
                "allgather_bucket_size": 5e8,
            },
            "activation_checkpointing": {
                "partition_activations": False,
                "cpu_checkpointing": False,
                "contiguous_memory_optimization": False,
                "number_checkpoints": 4,
                "synchronize_checkpoint_boundary": False,
                "profile": False,
            },
            "wall_clock_breakdown": False,
        }


class DetectionLoss(nn.Module):
    """DETR-style detection loss."""

    def __init__(
        self,
        num_classes,
        bbox_loss_coef=5.0,
        giou_loss_coef=2.0,
        class_loss_coef=1.0,
        focal_alpha=0.25,
        focal_gamma=2.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.bbox_loss_coef = bbox_loss_coef
        self.giou_loss_coef = giou_loss_coef
        self.class_loss_coef = class_loss_coef
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def forward(self, outputs, targets):
        """Compute detection loss."""
        pred_logits = outputs["pred_logits"]
        pred_boxes = outputs["pred_boxes"]

        # For simplicity, using basic L1 loss for bounding boxes
        # In practice, you'd implement Hungarian matching and GIoU loss
        loss_dict = {}

        # Classification loss (focal loss)
        target_classes = targets["labels"]
        loss_ce = nn.functional.cross_entropy(
            pred_logits.flatten(0, 1), target_classes.flatten(0, 1), reduction="mean"
        )
        loss_dict["loss_ce"] = loss_ce * self.class_loss_coef

        # Bounding box loss (simplified)
        target_boxes = targets["boxes"]
        loss_bbox = nn.functional.l1_loss(pred_boxes, target_boxes, reduction="mean")
        loss_dict["loss_bbox"] = loss_bbox * self.bbox_loss_coef

        # Total loss
        total_loss = sum(loss_dict.values())
        loss_dict["loss"] = total_loss

        return loss_dict


def train_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    scaler,
    config,
    model_engine=None,
    is_deepspeed=False,
):
    """Train for one epoch."""
    if is_deepspeed:
        model_engine.train()
    else:
        model.train()

    loss_meter = AverageMeter()
    ce_meter = AverageMeter()
    bbox_meter = AverageMeter()

    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, batch in enumerate(pbar):
        if config["model"].get("multi_scale", False):
            images = [img.to(device) for img in batch["images"]]
            targets = {
                "labels": batch["labels"].to(device),
                "boxes": batch["boxes"].to(device),
            }
        else:
            images = batch["images"].to(device)
            targets = {
                "labels": batch["labels"].to(device),
                "boxes": batch["boxes"].to(device),
            }

        if is_deepspeed:
            # DeepSpeed handles forward, backward, and optimization
            outputs = model_engine(images)
            loss_dict = criterion(outputs, targets)
            loss = loss_dict["loss"]

            model_engine.backward(loss)
            model_engine.step()
        else:
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(
                enabled=config["training"].get("mixed_precision", False)
            ):
                outputs = model(images)
                loss_dict = criterion(outputs, targets)
                loss = loss_dict["loss"]

            if config["training"].get("mixed_precision", False):
                scaler.scale(loss).backward()
                if config["training"].get("gradient_clip", 0) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config["training"]["gradient_clip"]
                    )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if config["training"].get("gradient_clip", 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config["training"]["gradient_clip"]
                    )
                optimizer.step()

        # Update meters
        loss_meter.update(loss.item())
        if "loss_ce" in loss_dict:
            ce_meter.update(loss_dict["loss_ce"].item())
        if "loss_bbox" in loss_dict:
            bbox_meter.update(loss_dict["loss_bbox"].item())

        # Update progress bar
        pbar.set_postfix(
            {
                "Loss": f"{loss_meter.avg:.4f}",
                "CE": f"{ce_meter.avg:.4f}",
                "BBox": f"{bbox_meter.avg:.4f}",
            }
        )

        if batch_idx % config["logging"]["log_interval"] == 0:
            logger.info(f"Batch {batch_idx}: Loss={loss_meter.avg:.4f}")

    return {
        "train_loss": loss_meter.avg,
        "train_ce_loss": ce_meter.avg,
        "train_bbox_loss": bbox_meter.avg,
    }


def validate(
    model, dataloader, criterion, device, config, model_engine=None, is_deepspeed=False
):
    """Validate model."""
    if is_deepspeed:
        model_engine.eval()
    else:
        model.eval()

    loss_meter = AverageMeter()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            if config["model"].get("multi_scale", False):
                images = [img.to(device) for img in batch["images"]]
                targets = {
                    "labels": batch["labels"].to(device),
                    "boxes": batch["boxes"].to(device),
                }
            else:
                images = batch["images"].to(device)
                targets = {
                    "labels": batch["labels"].to(device),
                    "boxes": batch["boxes"].to(device),
                }

            if is_deepspeed:
                outputs = model_engine(images)
            else:
                outputs = model(images)

            loss_dict = criterion(outputs, targets)
            loss_meter.update(loss_dict["loss"].item())

    return {"val_loss": loss_meter.avg}


def main():
    args = parse_args()

    # Initialize distributed training if enabled
    if args.distributed:
        if args.local_rank == -1:
            args.local_rank = int(os.environ.get("LOCAL_RANK", 0))

        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl")

        if args.world_size == -1:
            args.world_size = dist.get_world_size()

    # Load and optimize configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Apply dynamic memory optimization
    config = adjust_config_for_gpu_memory(config)

    set_seed(config.get("seed", 42))
    device = (
        get_device()
        if not args.distributed
        else torch.device(f"cuda:{args.local_rank}")
    )

    logger.info(f"Training {config['name']}")
    logger.info(f"Device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup experiment tracking (only on main process)
    if not args.distributed or args.local_rank == 0:
        tracker = ExperimentTracker(
            experiment_name=config["logging"]["experiment_name"],
            save_dir=config["logging"]["save_dir"],
        )
        tracker.log_config(config)
    else:
        tracker = None
        logger = None

    # Create datasets
    data_dir = args.data_dir or config["data"]["data_dir"]

    train_dataset = COCODataset(
        data_dir=data_dir,
        split=config["data"]["train_split"],
        image_size=config["data"]["image_size"],
        multi_scale=config["model"].get("multi_scale", False),
        scales=config["model"].get("scales", [800]),
        augment=True,
        debug=args.debug,
    )

    val_dataset = COCODataset(
        data_dir=data_dir,
        split=config["data"]["val_split"],
        image_size=config["data"]["image_size"],
        multi_scale=config["model"].get("multi_scale", False),
        scales=config["model"].get("scales", [800]),
        augment=False,
        debug=args.debug,
    )

    # Create distributed samplers if distributed
    train_sampler = None
    val_sampler = None
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config["data"]["num_workers"],
        pin_memory=config["data"]["pin_memory"],
        collate_fn=train_dataset.collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        sampler=val_sampler,
        num_workers=config["data"]["num_workers"],
        pin_memory=config["data"]["pin_memory"],
        collate_fn=val_dataset.collate_fn,
    )

    if logger:
        logger.info(f"Train dataset: {len(train_dataset)} samples")
        logger.info(f"Val dataset: {len(val_dataset)} samples")

    # Create model
    model_config = config["model"].copy()
    model_config.pop("name")  # Remove name from config

    model = create_rat_detection_model(**model_config)

    # Initialize DeepSpeed or regular distributed training
    model_engine = None
    optimizer = None
    scheduler = None

    if args.deepspeed and DEEPSPEED_AVAILABLE:
        if logger:
            logger.info(f"Using DeepSpeed with ZeRO Stage {args.zero_stage}")

        # Create DeepSpeed configuration
        deepspeed_config = create_deepspeed_config(
            zero_stage=args.zero_stage,
            train_micro_batch_size=config["data"]["batch_size"],
        )

        # Initialize DeepSpeed engine
        model_engine, optimizer, _, scheduler = deepspeed.initialize(
            model=model,
            config=deepspeed_config,
            dist_init_required=not args.distributed,
        )

        device = model_engine.device
        is_deepspeed = True

        if logger:
            logger.info(f"DeepSpeed initialized with device: {device}")

    else:
        # Standard training setup
        model = model.to(device)
        is_deepspeed = False

        if args.distributed:
            model = DDP(model, device_ids=[args.local_rank])

        # Optimizer
        optimizer_config = config["training"]["optimizer"]
        if optimizer_config["name"] == "adamw":
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=optimizer_config["lr"],
                weight_decay=optimizer_config["weight_decay"],
                betas=optimizer_config["betas"],
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_config['name']}")

        # Scheduler
        scheduler_config = config["training"].get("scheduler", {})
        if scheduler_config.get("name") == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=scheduler_config["step_size"],
                gamma=scheduler_config["gamma"],
            )
        elif scheduler_config.get("name") == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config["training"]["num_epochs"],
                eta_min=scheduler_config.get("min_lr", 0),
            )
        else:
            scheduler = None

    if logger:
        if is_deepspeed:
            # For DeepSpeed, parameter counting is more complex due to partitioning
            logger.info(
                "Model parameters: DeepSpeed partitioned (exact count not available)"
            )
        else:
            logger.info(
                f"Model parameters: "
                f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
            )

    # Loss function
    criterion = DetectionLoss(
        num_classes=config["model"]["num_classes"],
        bbox_loss_coef=config["model"]["bbox_loss_coef"],
        giou_loss_coef=config["model"]["giou_loss_coef"],
        class_loss_coef=config["model"]["class_loss_coef"],
        focal_alpha=config["model"]["focal_alpha"],
        focal_gamma=config["model"]["focal_gamma"],
    )

    # Mixed precision scaler (not used with DeepSpeed)
    scaler = None
    if not is_deepspeed and config["training"].get("mixed_precision", False):
        scaler = torch.cuda.amp.GradScaler()

    # Training loop
    best_val_loss = float("inf")
    start_epoch = 0

    # Resume if specified (simplified for DeepSpeed)
    if args.resume and not is_deepspeed:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]
        if logger:
            logger.info(f"Resumed from epoch {start_epoch}")

    if logger:
        logger.info(
            f"Starting training for {config['training']['num_epochs']} epochs..."
        )

    for epoch in range(start_epoch, config["training"]["num_epochs"]):
        if logger:
            logger.info(f"\nEpoch {epoch + 1}/{config['training']['num_epochs']}")
            logger.info("-" * 50)

        # Update sampler epoch for distributed training
        if args.distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Training
        train_metrics = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            config,
            model_engine=model_engine,
            is_deepspeed=is_deepspeed,
        )

        # Validation
        if (epoch + 1) % config["eval"]["eval_interval"] == 0:
            val_metrics = validate(
                model,
                val_loader,
                criterion,
                device,
                config,
                model_engine=model_engine,
                is_deepspeed=is_deepspeed,
            )

            # Log metrics (only on main process)
            if tracker is not None:
                all_metrics = {**train_metrics, **val_metrics}
                tracker.log_metrics(all_metrics, epoch)

            if logger:
                logger.info(f"Train Loss: {train_metrics['train_loss']:.4f}")
                logger.info(f"Val Loss: {val_metrics['val_loss']:.4f}")

            # Save best model (only on main process)
            if (not args.distributed or args.local_rank == 0) and val_metrics[
                "val_loss"
            ] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]

                if is_deepspeed:
                    # DeepSpeed model saving
                    model_engine.save_checkpoint(output_dir, tag=f"best_epoch_{epoch}")
                else:
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": (
                                scheduler.state_dict() if scheduler else None
                            ),
                            "best_val_loss": best_val_loss,
                            "config": config,
                        },
                        output_dir / "best_model.pth",
                    )

                logger.info(f"New best model saved! Val Loss: {best_val_loss:.4f}")

        # Update scheduler (not for DeepSpeed as it handles scheduling internally)
        if scheduler and not is_deepspeed:
            scheduler.step()

        # Save regular checkpoint (only on main process)
        if (not args.distributed or args.local_rank == 0) and (epoch + 1) % 10 == 0:
            if is_deepspeed:
                model_engine.save_checkpoint(output_dir, tag=f"epoch_{epoch + 1}")
            else:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": (
                            scheduler.state_dict() if scheduler else None
                        ),
                        "best_val_loss": best_val_loss,
                        "config": config,
                    },
                    output_dir / f"checkpoint_epoch_{epoch + 1}.pth",
                )

    if not args.distributed or args.local_rank == 0:
        logger.info(f"\nTraining completed! Best validation loss: {best_val_loss:.4f}")
        if tracker is not None:
            tracker.finish()

    # Clean up distributed processes
    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
