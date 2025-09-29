#!/usr/bin/env python3
"""
Simple training script that ACTUALLY WORKS.
No Ray Train, no DeepSpeed complexity, just training.
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import argparse
from pathlib import Path
from datetime import datetime

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


def setup_distributed():
    """Simple distributed setup without the complexity"""
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])

        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

        return rank, world_size, local_rank
    else:
        return 0, 1, 0


def create_simple_model(config):
    """Create a simple segmentation model"""
    print("Creating simple model...")

    # Use a simple U-Net style model instead of RAT for now
    class SimpleUNet(nn.Module):
        def __init__(self, in_channels=3, out_channels=1):
            super().__init__()

            # Encoder
            self.enc1 = nn.Sequential(
                nn.Conv2d(in_channels, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )

            self.pool1 = nn.MaxPool2d(2)

            self.enc2 = nn.Sequential(
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            )

            # Decoder
            self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
            self.dec2 = nn.Sequential(
                nn.Conv2d(128, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )

            self.final = nn.Conv2d(64, out_channels, 1)

        def forward(self, x):
            # Encoder
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool1(e1))

            # Decoder
            d2 = self.up2(e2)
            d2 = torch.cat([e1, d2], dim=1)
            d2 = self.dec2(d2)

            return torch.sigmoid(self.final(d2))

    return SimpleUNet(
        in_channels=config["model"]["input_features"],
        out_channels=config["model"]["num_classes"],
    )


def create_synthetic_dataset(config):
    """Create synthetic data for immediate testing"""
    from torch.utils.data import Dataset
    import torch

    class SyntheticDataset(Dataset):
        def __init__(self, size=100, image_size=256):
            self.size = size
            self.image_size = image_size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            # Random image
            image = torch.randn(3, self.image_size, self.image_size)

            # Random binary mask
            mask = torch.randint(0, 2, (1, self.image_size, self.image_size)).float()

            return image, mask

    return SyntheticDataset(size=200, image_size=config["data"]["image_size"])


def simple_train():
    """Simple training function that works"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    # Setup distributed
    rank, world_size, local_rank = setup_distributed()

    if rank == 0:
        print(f"🚀 Starting SIMPLE training (no Ray Train BS)")
        print(f"World size: {world_size}, Local rank: {local_rank}")

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Create model
    model = create_simple_model(config)
    model = model.cuda()

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    # Create dataset
    dataset = create_synthetic_dataset(config)

    # Create dataloader
    sampler = DistributedSampler(dataset) if world_size > 1 else None
    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=2,
        pin_memory=True,
    )

    # Setup training
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(config["training"]["learning_rate"])
    )
    criterion = nn.BCELoss()

    # Training loop
    model.train()

    for epoch in range(config["training"]["epochs"]):
        if sampler:
            sampler.set_epoch(epoch)

        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (images, masks) in enumerate(dataloader):
            images = images.cuda(non_blocking=True)
            masks = masks.cuda(non_blocking=True)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, masks)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            if batch_idx % 10 == 0 and rank == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / num_batches

        if rank == 0:
            print(f"✅ Epoch {epoch} completed! Average loss: {avg_loss:.4f}")

            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                checkpoint_dir = Path("results/simple_training/checkpoints")
                checkpoint_dir.mkdir(parents=True, exist_ok=True)

                model_state = (
                    model.module.state_dict()
                    if hasattr(model, "module")
                    else model.state_dict()
                )
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model_state,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": avg_loss,
                    },
                    checkpoint_dir / f"checkpoint_epoch_{epoch}.pth",
                )

                print(f"💾 Checkpoint saved at epoch {epoch}")

    if rank == 0:
        print("🎉 Training completed successfully!")
        print("This proves the basic infrastructure works!")
        print("Now we can debug the RAT model separately.")


if __name__ == "__main__":
    simple_train()
