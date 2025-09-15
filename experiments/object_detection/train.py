"""Training script for object detection experiments."""

import argparse
from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

# Add common utilities to path
sys.path.append(str(Path(__file__).parent.parent / "common"))
from datasets import COCODataset
from models import create_rat_detection_model
from utils import AverageMeter, ExperimentTracker, get_device, set_seed


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
    return parser.parse_args()


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


def train_epoch(model, dataloader, criterion, optimizer, device, scaler, config):
    """Train for one epoch."""
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
            print(f"Batch {batch_idx}: Loss={loss_meter.avg:.4f}")

    return {
        "train_loss": loss_meter.avg,
        "train_ce_loss": ce_meter.avg,
        "train_bbox_loss": bbox_meter.avg,
    }


def validate(model, dataloader, criterion, device, config):
    """Validate model."""
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

            outputs = model(images)
            loss_dict = criterion(outputs, targets)
            loss_meter.update(loss_dict["loss"].item())

    return {"val_loss": loss_meter.avg}


def main():
    args = parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    set_seed(config.get("seed", 42))
    device = get_device()

    print(f"Training {config['name']}")
    print(f"Device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup experiment tracking
    tracker = ExperimentTracker(
        name=config["logging"]["experiment_name"],
        save_dir=config["logging"]["save_dir"],
        config=config,
    )

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        pin_memory=config["data"]["pin_memory"],
        collate_fn=train_dataset.collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=config["data"]["pin_memory"],
        collate_fn=val_dataset.collate_fn,
    )

    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")

    # Create model
    model_config = config["model"].copy()
    model_config.pop("name")  # Remove name from config

    model = create_rat_detection_model(**model_config)
    model = model.to(device)

    print(
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

    # Mixed precision scaler
    scaler = (
        torch.cuda.amp.GradScaler()
        if config["training"].get("mixed_precision", False)
        else None
    )

    # Training loop
    best_val_loss = float("inf")
    start_epoch = 0

    # Resume if specified
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]
        print(f"Resumed from epoch {start_epoch}")

    print(f"Starting training for {config['training']['num_epochs']} epochs...")

    for epoch in range(start_epoch, config["training"]["num_epochs"]):
        print(f"\nEpoch {epoch + 1}/{config['training']['num_epochs']}")
        print("-" * 50)

        # Training
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler, config
        )

        # Validation
        if (epoch + 1) % config["eval"]["eval_interval"] == 0:
            val_metrics = validate(model, val_loader, criterion, device, config)

            # Log metrics
            all_metrics = {**train_metrics, **val_metrics}
            tracker.log_metrics(all_metrics, epoch)

            print(f"Train Loss: {train_metrics['train_loss']:.4f}")
            print(f"Val Loss: {val_metrics['val_loss']:.4f}")

            # Save best model
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]

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

                print(f"New best model saved! Val Loss: {best_val_loss:.4f}")

        # Update scheduler
        if scheduler:
            scheduler.step()

        # Save regular checkpoint
        if (epoch + 1) % 10 == 0:
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

    print(f"\nTraining completed! Best validation loss: {best_val_loss:.4f}")
    tracker.finish()


if __name__ == "__main__":
    main()
