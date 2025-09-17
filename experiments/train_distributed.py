#!/usr/bin/env python3
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
