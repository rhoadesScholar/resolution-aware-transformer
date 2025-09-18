#!/usr/bin/env python3
"""
Unified training script for RAT experiments.
Supports both medical segmentation and object detection tasks.
"""

import os
import sys
import argparse
from pathlib import Path
import yaml

# Add the experiments and core directories to Python path
EXPERIMENTS_DIR = Path(__file__).parent
CORE_DIR = EXPERIMENTS_DIR / "core"
COMMON_DIR = EXPERIMENTS_DIR / "common"

sys.path.insert(0, str(CORE_DIR))
sys.path.insert(0, str(COMMON_DIR))

# Import unified training framework
from trainer import UnifiedTrainer, train_segmentation_model, train_detection_model
from config import ConfigManager, load_config_with_auto_optimization

# Import task-specific components
try:
    from models import create_model, get_model_config
    from datasets import ISICDataset, COCODataset
    from utils import set_seed, setup_logging
except ImportError as e:
    print(f"Warning: Could not import common components: {e}")
    print("Make sure you're running from the experiments directory")
    sys.exit(1)

# Import model creation functions
try:
    from resolution_aware_transformer import ResolutionAwareTransformer
    RAT_AVAILABLE = True
except ImportError:
    print("Warning: RAT not available. Using placeholder.")
    RAT_AVAILABLE = False


class SegmentationLoss:
    """Loss function for segmentation tasks."""
    
    def __init__(self, loss_type: str = "bce"):
        import torch.nn as nn
        import torch
        
        self.loss_type = loss_type
        
        if loss_type == "bce":
            self.criterion = nn.BCEWithLogitsLoss()
        elif loss_type == "dice":
            self.criterion = self._dice_loss
        elif loss_type == "combined":
            self.bce_loss = nn.BCEWithLogitsLoss()
            self.criterion = self._combined_loss
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def _dice_loss(self, pred, target, smooth=1e-6):
        import torch
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
        
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()
    
    def _combined_loss(self, pred, target):
        bce = self.bce_loss(pred, target)
        dice = self._dice_loss(pred, target)
        return 0.5 * bce + 0.5 * dice
    
    def __call__(self, pred, target):
        return self.criterion(pred, target)


class DetectionLoss:
    """Loss function for detection tasks."""
    
    def __init__(self, num_classes: int = 80):
        import torch.nn as nn
        
        self.num_classes = num_classes
        self.ce_loss = nn.CrossEntropyLoss()
        self.bbox_loss = nn.L1Loss()
        self.class_weight = 1.0
        self.bbox_weight = 5.0
    
    def __call__(self, outputs, targets):
        import torch
        
        if isinstance(outputs, dict):
            # DETR-style outputs
            pred_logits = outputs.get("pred_logits")
            pred_boxes = outputs.get("pred_boxes")
            
            target_labels = targets.get("labels")
            target_boxes = targets.get("boxes")
            
            total_loss = 0
            
            if pred_logits is not None and target_labels is not None:
                class_loss = self.ce_loss(pred_logits.flatten(0, 1), target_labels.flatten())
                total_loss += self.class_weight * class_loss
            
            if pred_boxes is not None and target_boxes is not None:
                bbox_loss = self.bbox_loss(pred_boxes, target_boxes)
                total_loss += self.bbox_weight * bbox_loss
            
            return total_loss
        
        else:
            # Simple classification case
            return self.ce_loss(outputs, targets)


def create_rat_model(config: dict):
    """Create RAT model based on configuration."""
    if not RAT_AVAILABLE:
        # Create a simple placeholder model for testing
        import torch.nn as nn
        import torch
        
        class PlaceholderModel(nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
                self.conv = nn.Conv2d(3, 64, 3, padding=1)
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                self.fc = nn.Linear(64, 1)  # Binary segmentation
            
            def forward(self, x):
                if isinstance(x, list):
                    x = x[0]  # Use first scale for multi-scale
                x = self.conv(x)
                x = self.pool(x)
                x = torch.flatten(x, 1)
                x = self.fc(x)
                # Expand to match input spatial dimensions
                return x.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 256, 256)
        
        return PlaceholderModel()
    
    # Create actual RAT model
    model_config = config["model"].copy()
    model_config.pop("name", None)  # Remove name from config
    
    return ResolutionAwareTransformer(**model_config)


def create_datasets(config: dict, data_dir: str, debug: bool = False):
    """Create train and validation datasets."""
    task_type = config.get("task_type", "segmentation")
    data_config = config.get("data", {})
    
    if task_type == "segmentation":
        # Create ISIC segmentation datasets
        train_dataset = ISICDataset(
            data_dir=data_dir,
            split="train",
            image_size=data_config.get("image_size", 256),
            multi_scale=config["model"].get("multi_scale", False),
            scales=config["model"].get("scales", [256, 128, 64]),
            debug=debug,
        )
        
        val_dataset = ISICDataset(
            data_dir=data_dir,
            split="val",
            image_size=data_config.get("image_size", 256),
            multi_scale=config["model"].get("multi_scale", False),
            scales=config["model"].get("scales", [256, 128, 64]),
            debug=debug,
        )
    
    elif task_type == "detection":
        # Create COCO detection datasets
        train_dataset = COCODataset(
            data_dir=data_dir,
            split=data_config.get("train_split", "train2017"),
            image_size=data_config.get("image_size", 800),
            multi_scale=config["model"].get("multi_scale", False),
            scales=config["model"].get("scales", [800, 600, 400]),
            augment=True,
            debug=debug,
        )
        
        val_dataset = COCODataset(
            data_dir=data_dir,
            split=data_config.get("val_split", "val2017"),
            image_size=data_config.get("image_size", 800),
            multi_scale=config["model"].get("multi_scale", False),
            scales=config["model"].get("scales", [800, 600, 400]),
            augment=False,
            debug=debug,
        )
    
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    
    return train_dataset, val_dataset


def main():
    parser = argparse.ArgumentParser(description="Unified RAT training script")
    
    # Configuration
    parser.add_argument("--config", type=str, required=True,
                       help="Path to experiment configuration file")
    parser.add_argument("--task-type", type=str, 
                       choices=["segmentation", "detection"],
                       help="Task type (overrides config)")
    
    # Data
    parser.add_argument("--data-dir", type=str,
                       help="Path to dataset (overrides config)")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="./outputs",
                       help="Output directory for checkpoints and logs")
    
    # Training control
    parser.add_argument("--resume", type=str,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode (smaller dataset)")
    
    # Override config options
    parser.add_argument("--batch-size", type=int,
                       help="Override batch size from config")
    parser.add_argument("--learning-rate", type=float,
                       help="Override learning rate from config")
    parser.add_argument("--epochs", type=int,
                       help="Override number of epochs from config")
    
    args = parser.parse_args()
    
    # Load and optimize configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config_with_auto_optimization(args.config)
    
    # Apply command line overrides
    if args.task_type:
        config["task_type"] = args.task_type
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.learning_rate:
        config["training"]["learning_rate"] = args.learning_rate
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    
    # Set up experiment
    experiment_name = config.get("experiment_name", "unified_experiment")
    task_type = config.get("task_type", "segmentation")
    
    print(f"Starting experiment: {experiment_name}")
    print(f"Task type: {task_type}")
    print(f"Configuration: {config}")
    
    # Set random seed
    seed = config.get("seed", 42)
    set_seed(seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(str(output_dir), experiment_name)
    logger.info(f"Starting {task_type} experiment: {experiment_name}")
    
    # Create datasets
    data_dir = args.data_dir or config["data"].get("data_dir")
    if not data_dir:
        raise ValueError("Data directory must be specified in config or via --data-dir")
    
    print(f"Loading datasets from: {data_dir}")
    train_dataset, val_dataset = create_datasets(config, data_dir, args.debug)
    
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")
    
    # Create model
    print("Creating model...")
    model = create_rat_model(config)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Create loss function
    if task_type == "segmentation":
        loss_type = config.get("training", {}).get("loss", "bce")
        criterion = SegmentationLoss(loss_type)
    elif task_type == "detection":
        num_classes = config.get("model", {}).get("num_classes", 80)
        criterion = DetectionLoss(num_classes)
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    
    print(f"Using loss function: {criterion}")
    
    # Create trainer
    print("Initializing trainer...")
    trainer = UnifiedTrainer(
        config=config,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        criterion=criterion,
        output_dir=output_dir,
        experiment_name=experiment_name,
    )
    
    # Load checkpoint if resuming
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Start training
    print("Starting training...")
    results = trainer.train()
    
    print("Training completed!")
    print(f"Best metric: {results.get('best_metric', 'N/A')}")
    print(f"Total time: {results.get('total_time', 'N/A'):.2f} seconds")
    
    # Save final results
    results_file = output_dir / "final_results.yaml"
    with open(results_file, 'w') as f:
        yaml.dump(results, f, default_flow_style=False)
    
    print(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()