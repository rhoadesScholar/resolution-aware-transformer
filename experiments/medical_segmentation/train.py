"""Training script for medical segmentation experiment."""

import argparse
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import yaml

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
    return parser.parse_args()


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
) -> dict:
    """Train for one epoch."""
    model.train()

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

    # Load configuration
    config = load_config(args.config)

    # Set up experiment
    set_seed(config.get("seed", 42))
    device = get_device()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(str(output_dir), "medical_segmentation_train")
    tracker = ExperimentTracker("medical_segmentation", str(output_dir))
    tracker.log_config(config)
    tracker.start_timer()

    # Initialize wandb if configured
    if config.get("wandb", {}).get("enabled", False):
        wandb.init(
            project=config["wandb"].get("project", "rat-medical-segmentation"),
            name=config["wandb"].get("name", "experiment"),
            config=config,
        )

    logger.info("Starting medical segmentation experiment")
    logger.info(f"Device: {device}")
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

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["data"].get("num_workers", 4),
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"] * 2,
        shuffle=False,
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

    model = model.to(device)
    logger.info(f"Model: {model}")
    logger.info(f"Parameters: {count_parameters(model):,}")

    # Create optimizer and scheduler
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
        logger.info(f"Epoch {epoch + 1}/{config['training']['epochs']}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch + 1, config
        )
        logger.info(
            f"Train - Loss: {train_metrics['loss']:.4f}, "
            f"Dice: {train_metrics['dice']:.4f}"
        )

        # Validate
        val_metrics = validate_epoch(
            model, val_loader, criterion, device, epoch + 1, config
        )
        logger.info(
            f"Val - Loss: {val_metrics['loss']:.4f}, Dice: {val_metrics['dice']:.4f}"
        )

        # Update scheduler
        if scheduler:
            scheduler.step()

        # Log metrics
        for key, value in train_metrics.items():
            tracker.log_metric(f"train_{key}", value, epoch)

        for key, value in val_metrics.items():
            tracker.log_metric(f"val_{key}", value, epoch)

        if config.get("wandb", {}).get("enabled", False):
            wandb.log(
                {
                    "epoch": epoch,
                    **{f"train_{k}": v for k, v in train_metrics.items()},
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                }
            )

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

    # Save final results
    tracker.log_metric("final_best_dice", best_dice)
    duration = tracker.end_timer()
    logger.info(f"Training completed in {duration:.2f} seconds")
    logger.info(f"Best validation Dice: {best_dice:.4f}")

    tracker.save_results()

    if config.get("wandb", {}).get("enabled", False):
        wandb.finish()


if __name__ == "__main__":
    main()
