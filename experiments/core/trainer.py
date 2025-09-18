"""Unified training framework for RAT experiments."""

import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
import logging

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from .utils import (
    auto_detect_distributed,
    calculate_optimal_batch_size,
    setup_accelerate,
    setup_deepspeed_stage2,
    setup_mlflow_tracking,
    get_available_memory,
    DEEPSPEED_AVAILABLE,
    ACCELERATE_AVAILABLE,
    MLFLOW_AVAILABLE,
)
from .config import ConfigManager


class UnifiedTrainer:
    """
    Unified training framework that handles:
    - Automatic distributed training detection and setup
    - DeepSpeed Stage 2 integration
    - Intelligent batch size selection
    - HuggingFace Accelerate integration
    - MLFlow experiment tracking
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        model: nn.Module,
        train_dataset: torch.utils.data.Dataset,
        val_dataset: torch.utils.data.Dataset,
        criterion: nn.Module,
        output_dir: Union[str, Path],
        experiment_name: Optional[str] = None,
    ):
        """
        Initialize unified trainer.
        
        Args:
            config: Training configuration dictionary
            model: PyTorch model to train
            train_dataset: Training dataset
            val_dataset: Validation dataset
            criterion: Loss function
            output_dir: Directory for saving outputs
            experiment_name: Name for experiment tracking
        """
        self.config = config
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.criterion = criterion
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name or config.get("experiment_name", "unnamed_experiment")
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Auto-detect distributed environment
        self.distributed_info = auto_detect_distributed()
        self.is_distributed = self.distributed_info["is_distributed"]
        self.rank = self.distributed_info["rank"]
        self.local_rank = self.distributed_info["local_rank"]
        self.world_size = self.distributed_info["world_size"]
        
        # Only log on rank 0
        self.is_main_process = self.rank == 0
        
        if self.is_main_process:
            self.logger.info(f"Initializing UnifiedTrainer for: {self.experiment_name}")
            self.logger.info(f"Distributed training: {self.is_distributed}")
            if self.is_distributed:
                self.logger.info(f"World size: {self.world_size}, Rank: {self.rank}")
        
        # Initialize training components
        self.device = None
        self.accelerator = None
        self.deepspeed_engine = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        
        # Training state
        self.current_epoch = 0
        self.best_metric = float('inf')  # Assuming loss minimization
        self.training_stats = []
        
        self._setup_training()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the trainer."""
        logger = logging.getLogger(f"UnifiedTrainer_{self.experiment_name}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _setup_training(self):
        """Setup all training components."""
        self._setup_device_and_distributed()
        self._setup_model_and_optimization()
        self._setup_data_loaders()
        self._setup_experiment_tracking()
    
    def _setup_device_and_distributed(self):
        """Setup device and distributed training."""
        if self.is_distributed:
            # Initialize distributed training
            if not dist.is_initialized():
                dist.init_process_group(
                    backend=self.distributed_info["backend"],
                    rank=self.rank,
                    world_size=self.world_size,
                )
            
            # Set device
            if torch.cuda.is_available():
                self.device = torch.device(f"cuda:{self.local_rank}")
                torch.cuda.set_device(self.local_rank)
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.is_main_process:
            self.logger.info(f"Using device: {self.device}")
    
    def _setup_model_and_optimization(self):
        """Setup model, optimizer, and training components."""
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup training backend (Accelerate, DeepSpeed, or standard)
        training_config = self.config.get("training", {})
        
        # Try Accelerate first if available and requested
        if ACCELERATE_AVAILABLE and training_config.get("use_accelerate", True):
            self.accelerator = setup_accelerate(training_config)
            if self.accelerator is not None:
                self._setup_with_accelerate()
                return
        
        # Try DeepSpeed if available and requested
        if DEEPSPEED_AVAILABLE and training_config.get("deepspeed", False):
            self._setup_with_deepspeed()
            return
        
        # Fallback to standard training
        self._setup_standard_training()
    
    def _setup_with_accelerate(self):
        """Setup training with HuggingFace Accelerate."""
        if self.is_main_process:
            self.logger.info("Setting up training with HuggingFace Accelerate")
        
        # Create optimizer
        training_config = self.config.get("training", {})
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=training_config.get("learning_rate", 1e-4),
            weight_decay=training_config.get("weight_decay", 0.01),
        )
        
        # Create scheduler
        if training_config.get("scheduler") == "cosine":
            num_epochs = training_config.get("epochs", 100)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=num_epochs
            )
        
        # Let Accelerate handle everything
        self.model, self.optimizer, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.scheduler
        )
        
        self.device = self.accelerator.device
    
    def _setup_with_deepspeed(self):
        """Setup training with DeepSpeed."""
        if self.is_main_process:
            self.logger.info("Setting up training with DeepSpeed Stage 2")
        
        # Create DeepSpeed config
        training_config = self.config.get("training", {})
        ds_config = setup_deepspeed_stage2(training_config, self.model, self.world_size)
        
        # Initialize DeepSpeed
        import deepspeed
        
        self.deepspeed_engine, self.optimizer, _, self.scheduler = deepspeed.initialize(
            model=self.model,
            config=ds_config,
        )
        
        self.model = self.deepspeed_engine
        self.device = self.deepspeed_engine.device
    
    def _setup_standard_training(self):
        """Setup standard PyTorch training."""
        if self.is_main_process:
            self.logger.info("Setting up standard PyTorch training")
        
        # Wrap with DDP if distributed
        if self.is_distributed:
            self.model = DDP(self.model, device_ids=[self.local_rank])
        
        # Create optimizer
        training_config = self.config.get("training", {})
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=training_config.get("learning_rate", 1e-4),
            weight_decay=training_config.get("weight_decay", 0.01),
        )
        
        # Create scheduler
        if training_config.get("scheduler") == "cosine":
            num_epochs = training_config.get("epochs", 100)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=num_epochs
            )
        
        # Mixed precision scaler
        if training_config.get("mixed_precision", False):
            self.scaler = torch.cuda.amp.GradScaler()
    
    def _setup_data_loaders(self):
        """Setup data loaders with distributed support."""
        training_config = self.config.get("training", {})
        data_config = self.config.get("data", {})
        
        batch_size = training_config.get("batch_size", 4)
        num_workers = data_config.get("num_workers", 4)
        pin_memory = data_config.get("pin_memory", True)
        
        # Auto-adjust batch size if needed
        if training_config.get("auto_batch_size", False):
            # Get sample input shape
            sample_data = self.train_dataset[0]
            if isinstance(sample_data, dict):
                sample_input = sample_data.get("image", sample_data.get("images"))
            else:
                sample_input = sample_data[0] if isinstance(sample_data, (list, tuple)) else sample_data
            
            if isinstance(sample_input, (list, tuple)):
                input_shape = sample_input[0].shape  # Multi-scale case
            else:
                input_shape = sample_input.shape
            
            memory_info = get_available_memory()
            gpu_memory = memory_info["gpu_memory_gb"] / max(1, self.world_size)  # Per-GPU memory
            
            optimal_batch_size = calculate_optimal_batch_size(
                self.model, input_shape[1:], gpu_memory  # Remove batch dimension
            )
            
            if optimal_batch_size != batch_size:
                if self.is_main_process:
                    self.logger.info(f"Auto-adjusted batch size from {batch_size} to {optimal_batch_size}")
                batch_size = optimal_batch_size
        
        # Create samplers for distributed training
        train_sampler = None
        val_sampler = None
        
        if self.is_distributed and self.accelerator is None:  # Accelerate handles this internally
            train_sampler = DistributedSampler(
                self.train_dataset, rank=self.rank, num_replicas=self.world_size
            )
            val_sampler = DistributedSampler(
                self.val_dataset, rank=self.rank, num_replicas=self.world_size, shuffle=False
            )
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size * 2,  # Larger batch for validation
            shuffle=False,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
        
        # Prepare with Accelerate if using it
        if self.accelerator is not None:
            self.train_loader, self.val_loader = self.accelerator.prepare(
                self.train_loader, self.val_loader
            )
        
        if self.is_main_process:
            self.logger.info(f"Created data loaders - Train: {len(self.train_loader)} batches, "
                           f"Val: {len(self.val_loader)} batches")
    
    def _setup_experiment_tracking(self):
        """Setup experiment tracking with MLFlow."""
        if self.is_main_process and MLFLOW_AVAILABLE:
            logging_config = self.config.get("logging", {})
            if logging_config.get("use_mlflow", False):
                setup_mlflow_tracking(self.config, self.experiment_name)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        # Set epoch for distributed sampler
        if self.is_distributed and hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(self.current_epoch)
        
        if self.is_main_process:
            pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1} [Train]")
        else:
            pbar = self.train_loader
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            if self.accelerator is None:
                if isinstance(batch, dict):
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                else:
                    batch = [item.to(self.device) if isinstance(item, torch.Tensor) else item 
                            for item in batch]
            
            # Forward pass and loss calculation
            if self.deepspeed_engine is not None:
                # DeepSpeed training
                loss = self._forward_pass_deepspeed(batch)
                self.deepspeed_engine.backward(loss)
                self.deepspeed_engine.step()
            elif self.accelerator is not None:
                # Accelerate training
                loss = self._forward_pass_accelerate(batch)
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.optimizer.zero_grad()
            else:
                # Standard training
                loss = self._forward_pass_standard(batch)
            
            total_loss += loss.item()
            
            # Update progress bar
            if self.is_main_process:
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / num_batches
        return {"loss": avg_loss}
    
    def _forward_pass_deepspeed(self, batch) -> torch.Tensor:
        """Forward pass with DeepSpeed."""
        # Extract inputs and targets from batch
        inputs, targets = self._extract_inputs_targets(batch)
        
        # Forward pass
        outputs = self.deepspeed_engine(inputs)
        loss = self.criterion(outputs, targets)
        
        return loss
    
    def _forward_pass_accelerate(self, batch) -> torch.Tensor:
        """Forward pass with Accelerate."""
        # Extract inputs and targets from batch
        inputs, targets = self._extract_inputs_targets(batch)
        
        # Forward pass
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        
        return loss
    
    def _forward_pass_standard(self, batch) -> torch.Tensor:
        """Forward pass with standard PyTorch."""
        training_config = self.config.get("training", {})
        
        self.optimizer.zero_grad()
        
        # Extract inputs and targets from batch
        inputs, targets = self._extract_inputs_targets(batch)
        
        # Forward pass with optional mixed precision
        if self.scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            if training_config.get("grad_clip", 0) > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), training_config["grad_clip"]
                )
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if training_config.get("grad_clip", 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), training_config["grad_clip"]
                )
            
            self.optimizer.step()
        
        return loss
    
    def _extract_inputs_targets(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract inputs and targets from batch (task-specific)."""
        # This is a generic implementation - should be overridden for specific tasks
        if isinstance(batch, dict):
            if "images" in batch and "masks" in batch:
                # Segmentation task
                return batch["images"], batch["masks"]
            elif "image" in batch and "mask" in batch:
                # Single-scale segmentation
                return batch["image"], batch["mask"]
            elif "images" in batch and "labels" in batch:
                # Detection task
                return batch["images"], {"labels": batch["labels"], "boxes": batch["boxes"]}
            else:
                # Generic case - assume first item is input, second is target
                items = list(batch.values())
                return items[0], items[1]
        else:
            # Assume tuple/list format
            return batch[0], batch[1]
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        if self.is_main_process:
            pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch + 1} [Val]")
        else:
            pbar = self.val_loader
        
        with torch.no_grad():
            for batch in pbar:
                # Move data to device
                if self.accelerator is None:
                    if isinstance(batch, dict):
                        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                                for k, v in batch.items()}
                    else:
                        batch = [item.to(self.device) if isinstance(item, torch.Tensor) else item 
                                for item in batch]
                
                # Forward pass
                inputs, targets = self._extract_inputs_targets(batch)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                
                # Update progress bar
                if self.is_main_process:
                    pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / num_batches
        return {"loss": avg_loss}
    
    def train(self) -> Dict[str, Any]:
        """
        Main training loop.
        
        Returns:
            Training statistics and results
        """
        training_config = self.config.get("training", {})
        num_epochs = training_config.get("epochs", 100)
        save_freq = training_config.get("save_freq", 10)
        
        if self.is_main_process:
            self.logger.info(f"Starting training for {num_epochs} epochs")
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = self.validate_epoch()
            
            # Update scheduler
            if self.scheduler is not None and self.deepspeed_engine is None:
                self.scheduler.step()
            
            # Log metrics
            if self.is_main_process:
                epoch_stats = {
                    "epoch": epoch + 1,
                    "train_loss": train_metrics["loss"],
                    "val_loss": val_metrics["loss"],
                    "lr": self.optimizer.param_groups[0]["lr"] if self.optimizer else 0.0,
                }
                
                self.training_stats.append(epoch_stats)
                
                self.logger.info(
                    f"Epoch {epoch + 1}/{num_epochs} - "
                    f"Train Loss: {train_metrics['loss']:.4f}, "
                    f"Val Loss: {val_metrics['loss']:.4f}"
                )
                
                # Log to MLFlow if available
                if MLFLOW_AVAILABLE:
                    try:
                        import mlflow
                        mlflow.log_metrics({
                            "train_loss": train_metrics["loss"],
                            "val_loss": val_metrics["loss"],
                            "learning_rate": epoch_stats["lr"],
                        }, step=epoch)
                    except Exception as e:
                        self.logger.warning(f"MLFlow logging failed: {e}")
                
                # Save best model
                if val_metrics["loss"] < self.best_metric:
                    self.best_metric = val_metrics["loss"]
                    self.save_checkpoint("best_model.pth", epoch_stats)
                
                # Save periodic checkpoint
                if (epoch + 1) % save_freq == 0:
                    self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pth", epoch_stats)
        
        # Training completed
        total_time = time.time() - start_time
        
        if self.is_main_process:
            self.logger.info(f"Training completed in {total_time:.2f} seconds")
            self.logger.info(f"Best validation loss: {self.best_metric:.4f}")
            
            # Save final results
            results = {
                "best_metric": self.best_metric,
                "total_time": total_time,
                "training_stats": self.training_stats,
                "config": self.config,
            }
            
            import json
            with open(self.output_dir / "training_results.json", "w") as f:
                json.dump(results, f, indent=2)
            
            return results
        
        # Cleanup distributed training
        if self.is_distributed:
            dist.destroy_process_group()
        
        return {}
    
    def save_checkpoint(self, filename: str, metadata: Optional[Dict[str, Any]] = None):
        """Save model checkpoint."""
        if not self.is_main_process:
            return
        
        checkpoint_path = self.output_dir / filename
        
        if self.deepspeed_engine is not None:
            # DeepSpeed checkpoint
            self.deepspeed_engine.save_checkpoint(
                str(self.output_dir), tag=filename.replace(".pth", "")
            )
        else:
            # Standard checkpoint
            checkpoint = {
                "epoch": self.current_epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_metric": self.best_metric,
                "config": self.config,
            }
            
            if self.scheduler is not None:
                checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
            
            if metadata:
                checkpoint["metadata"] = metadata
            
            torch.save(checkpoint, checkpoint_path)
        
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]):
        """Load model checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        
        if self.deepspeed_engine is not None:
            # DeepSpeed checkpoint loading
            self.deepspeed_engine.load_checkpoint(str(checkpoint_path.parent), tag=checkpoint_path.stem)
        else:
            # Standard checkpoint loading
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
            if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            
            self.current_epoch = checkpoint.get("epoch", 0)
            self.best_metric = checkpoint.get("best_metric", float('inf'))
        
        if self.is_main_process:
            self.logger.info(f"Loaded checkpoint: {checkpoint_path}")


# Convenience functions for common training scenarios

def train_segmentation_model(
    config_path: Union[str, Path],
    model: nn.Module,
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    output_dir: Union[str, Path],
) -> Dict[str, Any]:
    """
    Convenience function for training segmentation models.
    
    Args:
        config_path: Path to configuration file
        model: Segmentation model
        train_dataset: Training dataset
        val_dataset: Validation dataset
        output_dir: Output directory
    
    Returns:
        Training results
    """
    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.load_config(config_path)
    
    # Create loss function (example for segmentation)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # Create trainer
    trainer = UnifiedTrainer(
        config=config,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        criterion=criterion,
        output_dir=output_dir,
        experiment_name=config.get("experiment_name", "segmentation_experiment"),
    )
    
    # Train
    return trainer.train()


def train_detection_model(
    config_path: Union[str, Path],
    model: nn.Module,
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    output_dir: Union[str, Path],
) -> Dict[str, Any]:
    """
    Convenience function for training detection models.
    
    Args:
        config_path: Path to configuration file
        model: Detection model
        train_dataset: Training dataset
        val_dataset: Validation dataset
        output_dir: Output directory
    
    Returns:
        Training results
    """
    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.load_config(config_path)
    
    # Create loss function (example for detection)
    class DetectionLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.ce_loss = nn.CrossEntropyLoss()
            self.bbox_loss = nn.L1Loss()
        
        def forward(self, outputs, targets):
            # Simplified detection loss
            if isinstance(outputs, dict):
                class_loss = self.ce_loss(outputs["pred_logits"], targets["labels"])
                bbox_loss = self.bbox_loss(outputs["pred_boxes"], targets["boxes"])
                return class_loss + bbox_loss
            else:
                return self.ce_loss(outputs, targets)
    
    criterion = DetectionLoss()
    
    # Create trainer
    trainer = UnifiedTrainer(
        config=config,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        criterion=criterion,
        output_dir=output_dir,
        experiment_name=config.get("experiment_name", "detection_experiment"),
    )
    
    # Train
    return trainer.train()