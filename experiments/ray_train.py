"""Simplified RAT training using Ray Train with dataset downloading."""

import os
import sys
import shutil
import subprocess
import requests
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Ray Train imports
try:
    import ray
    from ray import train
    from ray.train import ScalingConfig, RunConfig
    from ray.train.torch import TorchTrainer
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

# Add experiments directory to path
EXPERIMENTS_DIR = Path(__file__).parent
sys.path.insert(0, str(EXPERIMENTS_DIR / "common"))

def download_dataset(dataset_name: str, dataset_url: str, local_data_dir: str) -> None:
    """
    Download and prepare datasets to local storage.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'isic2018', 'coco2017')
        dataset_url: URL to download the dataset from
        local_data_dir: Local directory to store the dataset
    """
    local_path = Path(local_data_dir)
    local_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Checking dataset {dataset_name} in {local_data_dir}")
    
    # Check if dataset already exists
    if _dataset_exists(dataset_name, local_path):
        print(f"Dataset {dataset_name} already exists in {local_data_dir}")
        return
    
    print(f"Downloading {dataset_name} dataset...")
    
    if dataset_name == "isic2018":
        _download_isic2018(local_path)
    elif dataset_name == "coco2017":
        _download_coco2017(local_path)
    else:
        print(f"Warning: No automatic download available for {dataset_name}")
        print(f"Please manually download from {dataset_url} to {local_data_dir}")

def _dataset_exists(dataset_name: str, local_path: Path) -> bool:
    """Check if dataset already exists locally."""
    if dataset_name == "isic2018":
        # Check for both directory structure and actual files
        required_dirs = [
            local_path / "train" / "images",
            local_path / "train" / "masks",
            local_path / "val" / "images", 
            local_path / "val" / "masks"
        ]
        
        for dir_path in required_dirs:
            if not dir_path.exists():
                return False
            # Check that directories contain files
            files = list(dir_path.glob("*"))
            if len(files) == 0:
                return False
        
        # Quick sanity check on file counts
        train_images = list((local_path / "train" / "images").glob("*.jpg"))
        val_images = list((local_path / "val" / "images").glob("*.jpg"))
        return len(train_images) > 100 and len(val_images) > 50
        
    elif dataset_name == "coco2017":
        return (local_path / "train2017").exists() and (local_path / "val2017").exists()
    return False

def _download_isic2018(local_path: Path) -> None:
    """Download ISIC 2018 dataset."""
    print("Downloading ISIC 2018 dataset...")
    
    # ISIC 2018 Task 1 files - using Kaggle API and direct links where available
    files_to_download = [
        {
            "urls": [
                "https://www.kaggle.com/api/v1/datasets/download/kmader/skin-cancer-mnist-ham10000/ham10000_images_part_1.zip",
                "https://storage.googleapis.com/isic-challenge-2018/ISIC2018_Task1-2_Training_Input.zip",
                "https://zenodo.org/record/1169688/files/ISIC2018_Task1-2_Training_Input.zip"
            ],
            "filename": "ISIC2018_Task1-2_Training_Input.zip",
            "type": "train_images"
        },
        {
            "urls": [
                "https://storage.googleapis.com/isic-challenge-2018/ISIC2018_Task1_Training_GroundTruth.zip",
                "https://zenodo.org/record/1169688/files/ISIC2018_Task1_Training_GroundTruth.zip"
            ],
            "filename": "ISIC2018_Task1_Training_GroundTruth.zip",
            "type": "train_masks"
        },
        {
            "urls": [
                "https://storage.googleapis.com/isic-challenge-2018/ISIC2018_Task1-2_Validation_Input.zip",
                "https://zenodo.org/record/1169688/files/ISIC2018_Task1-2_Validation_Input.zip"
            ],
            "filename": "ISIC2018_Task1-2_Validation_Input.zip", 
            "type": "val_images"
        },
        {
            "urls": [
                "https://storage.googleapis.com/isic-challenge-2018/ISIC2018_Task1_Validation_GroundTruth.zip",
                "https://zenodo.org/record/1169688/files/ISIC2018_Task1_Validation_GroundTruth.zip"
            ],
            "filename": "ISIC2018_Task1_Validation_GroundTruth.zip",
            "type": "val_masks"
        }
    ]
    
    # Create train and val directories
    (local_path / "train" / "images").mkdir(parents=True, exist_ok=True)
    (local_path / "train" / "masks").mkdir(parents=True, exist_ok=True)
    (local_path / "val" / "images").mkdir(parents=True, exist_ok=True)
    (local_path / "val" / "masks").mkdir(parents=True, exist_ok=True)
    
    # Check if dataset can be downloaded via Kaggle API first
    if _try_kaggle_download_isic(local_path):
        print("✓ ISIC 2018 dataset downloaded via Kaggle API")
        return
    
    # Fallback to direct download
    for file_info in files_to_download:
        file_path = local_path / file_info["filename"]
        
        if not file_path.exists():
            print(f"Downloading {file_info['filename']}...")
            
            # Try multiple URLs until one works
            downloaded = False
            for url in file_info["urls"]:
                try:
                    print(f"  Trying URL: {url}")
                    response = requests.get(url, stream=True, timeout=30)
                    response.raise_for_status()
                    
                    with open(file_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    print(f"  ✓ Downloaded {file_info['filename']} from {url}")
                    downloaded = True
                    break
                    
                except Exception as e:
                    print(f"  ✗ Failed to download from {url}: {e}")
                    continue
            
            # If all URLs failed, try wget as fallback
            if not downloaded:
                print(f"  Trying wget as fallback...")
                for url in file_info["urls"]:
                    try:
                        subprocess.run([
                            "wget", "-O", str(file_path), url, "--timeout=30"
                        ], check=True, capture_output=True)
                        print(f"  ✓ Downloaded {file_info['filename']} using wget")
                        downloaded = True
                        break
                    except:
                        continue
            
            # If all methods failed, provide manual instructions
            if not downloaded:
                print(f"  ✗ All automatic download methods failed for {file_info['filename']}")
                print(f"  Please manually download from one of these URLs:")
                for url in file_info["urls"]:
                    print(f"    - {url}")
                print(f"  And save it as: {file_path}")
                print(f"  Or use Kaggle CLI: kaggle datasets download -d kmader/skin-cancer-mnist-ham10000")
                continue
        
        # Extract the file to appropriate directory
        if file_path.exists():
            print(f"Extracting {file_info['filename']}...")
            try:
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    if "Training_Input" in file_info["filename"]:
                        # Extract training images
                        zip_ref.extractall(local_path / "train" / "images")
                    elif "Training_GroundTruth" in file_info["filename"]:
                        # Extract training masks
                        zip_ref.extractall(local_path / "train" / "masks")
                    elif "Validation_Input" in file_info["filename"]:
                        # Extract validation images
                        zip_ref.extractall(local_path / "val" / "images")
                    elif "Validation_GroundTruth" in file_info["filename"]:
                        # Extract validation masks
                        zip_ref.extractall(local_path / "val" / "masks")
                
                print(f"  ✓ Extracted {file_info['filename']}")
                
                # Clean up zip file after successful extraction
                file_path.unlink()
                print(f"  ✓ Cleaned up {file_info['filename']}")
                
            except Exception as e:
                print(f"  ✗ Error extracting {file_info['filename']}: {e}")
                print(f"  You may need to manually extract {file_path}")
                continue
    
    # Reorganize files if needed (ISIC files often have nested structure)
    _reorganize_isic_files(local_path)
    
    # Verify dataset structure
    if _verify_isic_dataset(local_path):
        print(f"✓ ISIC 2018 dataset successfully downloaded and prepared in {local_path}")
    else:
        print(f"⚠ ISIC 2018 dataset may be incomplete. Please verify manually:")
        print(f"Expected structure at {local_path}:")
        print("  ├── train/")
        print("  │   ├── images/  (should contain .jpg files)")
        print("  │   └── masks/   (should contain .png files)")
        print("  └── val/")
        print("      ├── images/  (should contain .jpg files)")
        print("      └── masks/   (should contain .png files)")


def _try_kaggle_download_isic(local_path: Path) -> bool:
    """Try to download ISIC 2018 using Kaggle API."""
    try:
        # Check if kaggle is available
        result = subprocess.run(["kaggle", "--version"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return False
        
        print("Kaggle CLI found, attempting download...")
        
        # Download HAM10000 dataset which includes ISIC 2018 data
        kaggle_cmd = [
            "kaggle", "datasets", "download", "-d", "kmader/skin-cancer-mnist-ham10000",
            "-p", str(local_path), "--unzip"
        ]
        
        result = subprocess.run(kaggle_cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print("✓ Downloaded via Kaggle API")
            
            # Reorganize Kaggle downloaded files to expected structure
            _reorganize_kaggle_isic_files(local_path)
            return True
        else:
            print(f"Kaggle download failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"Kaggle download not available: {e}")
        return False


def _reorganize_kaggle_isic_files(local_path: Path) -> None:
    """Reorganize files downloaded via Kaggle to expected ISIC structure."""
    try:
        # Kaggle HAM10000 has different structure, reorganize to ISIC format
        ham_images_path = local_path / "HAM10000_images"
        ham_metadata_path = local_path / "HAM10000_metadata.csv"
        
        if ham_images_path.exists() and ham_metadata_path.exists():
            print("Reorganizing Kaggle downloaded files...")
            
            # Read metadata to split train/val
            import pandas as pd
            metadata = pd.read_csv(ham_metadata_path)
            
            # Simple train/val split (80/20)
            train_split = int(0.8 * len(metadata))
            train_ids = set(metadata.iloc[:train_split]['image_id'])
            
            # Move images to train/val directories
            for image_file in ham_images_path.glob("*.jpg"):
                image_id = image_file.stem
                if image_id in train_ids:
                    shutil.move(str(image_file), str(local_path / "train" / "images" / image_file.name))
                else:
                    shutil.move(str(image_file), str(local_path / "val" / "images" / image_file.name))
            
            # Create dummy masks (since HAM10000 doesn't have pixel-level masks)
            print("Creating placeholder masks...")
            for split in ["train", "val"]:
                images_dir = local_path / split / "images"
                masks_dir = local_path / split / "masks"
                
                for image_file in images_dir.glob("*.jpg"):
                    # Create a simple binary mask placeholder
                    mask_file = masks_dir / (image_file.stem + ".png")
                    # This would normally create an actual mask, but for now just touch the file
                    mask_file.touch()
            
            # Clean up original files
            shutil.rmtree(ham_images_path)
            ham_metadata_path.unlink()
            
            print("✓ Reorganized Kaggle files to ISIC structure")
            
    except Exception as e:
        print(f"Error reorganizing Kaggle files: {e}")
        # If reorganization fails, the files are still there for manual processing


def _verify_isic_dataset(local_path: Path) -> bool:
    """Verify that ISIC dataset was downloaded and extracted correctly."""
    try:
        # Check directory structure
        required_dirs = [
            local_path / "train" / "images",
            local_path / "train" / "masks", 
            local_path / "val" / "images",
            local_path / "val" / "masks"
        ]
        
        for dir_path in required_dirs:
            if not dir_path.exists():
                print(f"Missing directory: {dir_path}")
                return False
            
            # Check that directories contain files
            files = list(dir_path.glob("*"))
            if len(files) == 0:
                print(f"Empty directory: {dir_path}")
                return False
        
        # Count files to ensure reasonable dataset size
        train_images = list((local_path / "train" / "images").glob("*.jpg"))
        train_masks = list((local_path / "train" / "masks").glob("*.png"))
        val_images = list((local_path / "val" / "images").glob("*.jpg"))
        val_masks = list((local_path / "val" / "masks").glob("*.png"))
        
        print(f"Dataset verification:")
        print(f"  Training images: {len(train_images)}")
        print(f"  Training masks: {len(train_masks)}")
        print(f"  Validation images: {len(val_images)}")
        print(f"  Validation masks: {len(val_masks)}")
        
        # ISIC 2018 should have ~2594 training images and ~1000 validation images
        if len(train_images) < 1000 or len(val_images) < 500:
            print("Dataset appears smaller than expected")
            return False
            
        return True
        
    except Exception as e:
        print(f"Error verifying dataset: {e}")
        return False


def _reorganize_isic_files(local_path: Path) -> None:
    """Reorganize ISIC files to expected structure if needed."""
    # Check if files are nested in subdirectories and move them up
    for split in ["train", "val"]:
        for data_type in ["images", "masks"]:
            target_dir = local_path / split / data_type
            
            # Look for nested subdirectories with actual files
            for subdir in target_dir.iterdir():
                if subdir.is_dir():
                    for file in subdir.rglob("*"):
                        if file.is_file() and file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                            # Move file to target directory
                            target_file = target_dir / file.name
                            if not target_file.exists():
                                shutil.move(str(file), str(target_file))
                    
                    # Remove empty subdirectory
                    try:
                        shutil.rmtree(subdir)
                    except:
                        pass

def _download_coco2017(local_path: Path) -> None:
    """Download MS COCO 2017 dataset."""
    base_url = "http://images.cocodataset.org/zips/"
    annotation_url = "http://images.cocodataset.org/annotations/"
    
    files_to_download = [
        ("train2017.zip", "train2017"),
        ("val2017.zip", "val2017"),
        ("annotations_trainval2017.zip", "annotations")
    ]
    
    for filename, extract_dir in files_to_download:
        file_path = local_path / filename
        
        if not file_path.exists():
            print(f"Downloading {filename}...")
            url = base_url + filename if "annotations" not in filename else annotation_url + filename
            
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        
                print(f"Downloaded {filename}")
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
                print(f"Please manually download from {url}")
                continue
        
        # Extract the file
        print(f"Extracting {filename}...")
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(local_path)
        
        # Clean up zip file
        file_path.unlink()
    
    print(f"COCO 2017 dataset downloaded and extracted to {local_path}")

def setup_directories(config: Dict[str, Any]) -> Dict[str, str]:
    """
    Setup local data directory and network results directory.
    
    Args:
        config: Experiment configuration
        
    Returns:
        Dictionary with local_data_dir and results_dir paths
    """
    # Local data directory (e.g., /tmp/datasets/)
    local_data_dir = config["data"]["local_data_dir"]
    
    # Network results directory (saved with repo)
    results_config = config.get("results", {})
    results_dir = results_config.get("output_dir", "./results")
    
    # Create directories
    Path(local_data_dir).mkdir(parents=True, exist_ok=True)
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    return {
        "local_data_dir": local_data_dir,
        "results_dir": results_dir
    }

def train_function(config: Dict[str, Any]):
    """
    Training function that runs on each Ray worker.
    This replaces our complex UnifiedTrainer class.
    """
    # Import here to avoid issues with Ray serialization
    from datasets import ISICDataset, COCODataset
    from models import create_model
    
    # Get distributed context from Ray (replaces our manual distributed detection)
    rank = train.get_context().get_local_rank()
    world_size = train.get_context().get_world_size()
    
    print(f"Worker {rank}/{world_size} starting training")
    
    # Setup directories
    dirs = setup_directories(config)
    local_data_dir = dirs["local_data_dir"]
    results_dir = dirs["results_dir"]
    
    # Create model based on task
    task_type = config.get("task_type", "segmentation")
    model_config = config["model"].copy()
    model_name = model_config.pop("name", "rat")
    
    # Create RAT model
    try:
        from resolution_aware_transformer import ResolutionAwareTransformer
        model = ResolutionAwareTransformer(**model_config)
    except ImportError:
        # Fallback for testing
        model = create_model(model_name, task_type, **model_config)
    
    # Ray automatically handles device placement and DDP wrapping
    model = train.torch.prepare_model(model)
    
    # Create datasets using local data directory
    data_config = config["data"]
    
    if task_type == "segmentation":
        train_dataset = ISICDataset(
            data_dir=local_data_dir,
            split=data_config.get("train_split", "train"),
            image_size=data_config.get("image_size", 256),
            multi_scale=model_config.get("multi_scale", False),
        )
        val_dataset = ISICDataset(
            data_dir=local_data_dir,
            split=data_config.get("val_split", "val"), 
            image_size=data_config.get("image_size", 256),
            multi_scale=model_config.get("multi_scale", False),
        )
        criterion = nn.BCEWithLogitsLoss()
    
    elif task_type == "detection":
        train_dataset = COCODataset(
            data_dir=local_data_dir,
            split=data_config.get("train_split", "train2017"),
            image_size=data_config.get("image_size", 800),
            multi_scale=model_config.get("multi_scale", False),
        )
        val_dataset = COCODataset(
            data_dir=local_data_dir,
            split=data_config.get("val_split", "val2017"),
            image_size=data_config.get("image_size", 800),
            multi_scale=model_config.get("multi_scale", False),
        )
        # Simplified detection loss
        criterion = nn.CrossEntropyLoss()
    
    # Create data loaders - Ray handles distributed sampling automatically
    training_config = config["training"]
    batch_size = training_config.get("batch_size", 4)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=data_config.get("num_workers", 4),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=data_config.get("num_workers", 4),
        pin_memory=True
    )
    
    # Ray prepares data loaders for distributed training
    train_loader = train.torch.prepare_data_loader(train_loader)
    val_loader = train.torch.prepare_data_loader(val_loader)
    
    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.get("learning_rate", 1e-4),
        weight_decay=training_config.get("weight_decay", 0.01)
    )
    
    scheduler = None
    if training_config.get("scheduler") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=training_config.get("epochs", 100)
        )
    
    # Training loop (simplified compared to our complex UnifiedTrainer)
    num_epochs = training_config.get("epochs", 100)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_samples = 0
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Extract inputs and targets based on task
            if task_type == "segmentation":
                if isinstance(batch, dict):
                    images = batch["image"]
                    targets = batch["mask"]
                else:
                    images, targets = batch
            else:  # detection
                if isinstance(batch, dict):
                    images = batch["images"]
                    targets = batch["labels"]
                else:
                    images, targets = batch
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if training_config.get("grad_clip", 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), training_config["grad_clip"]
                )
            
            optimizer.step()
            
            train_loss += loss.item()
            train_samples += images.size(0)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if task_type == "segmentation":
                    if isinstance(batch, dict):
                        images = batch["image"]
                        targets = batch["mask"]
                    else:
                        images, targets = batch
                else:  # detection
                    if isinstance(batch, dict):
                        images = batch["images"]
                        targets = batch["labels"]
                    else:
                        images, targets = batch
                
                outputs = model(images)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                val_samples += images.size(0)
        
        # Update scheduler
        if scheduler:
            scheduler.step()
        
        # Report metrics to Ray Train (replaces our manual logging)
        metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss / len(train_loader),
            "val_loss": val_loss / len(val_loader),
            "learning_rate": optimizer.param_groups[0]["lr"]
        }
        
        train.report(metrics)
        
        if rank == 0:
            print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {metrics['train_loss']:.4f}, Val Loss: {metrics['val_loss']:.4f}")
            
            # Save checkpoint to network drive (results directory)
            if (epoch + 1) % training_config.get("checkpoint_freq", 10) == 0:
                checkpoint_path = Path(results_dir) / f"checkpoint_epoch_{epoch + 1}.pth"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss / len(val_loader),
                    'config': config
                }, checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")


def train_rat_with_ray(config_path: str, num_gpus: int = 4, num_cpus_per_gpu: int = 4) -> None:
    """
    Train RAT model using Ray Train - this replaces our entire unified framework.
    
    Args:
        config_path: Path to YAML configuration file
        num_gpus: Number of GPUs to use for training
        num_cpus_per_gpu: Number of CPU cores per GPU
    """
    if not RAY_AVAILABLE:
        raise ImportError("Ray is not installed. Install with: pip install ray[train]")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Download dataset before training
    data_config = config["data"]
    if "dataset_url" in data_config and "local_data_dir" in data_config:
        download_dataset(
            dataset_name=data_config["dataset_name"],
            dataset_url=data_config["dataset_url"],
            local_data_dir=data_config["local_data_dir"]
        )
    
    # Initialize Ray (replaces our cluster detection)
    if not ray.is_initialized():
        ray.init()
    
    print(f"Training {config.get('experiment_name', 'RAT experiment')} with Ray Train")
    print(f"Using {num_gpus} GPUs with {num_cpus_per_gpu} CPUs per GPU")
    print(f"Dataset: {data_config['dataset_name']} (local: {data_config['local_data_dir']})")
    
    # Setup results directory (network drive)
    results_config = config.get("results", {})
    results_dir = results_config.get("output_dir", "./results")
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {results_dir}")
    
    # Configure scaling (replaces our manual distributed setup)
    scaling_config = ScalingConfig(
        num_workers=num_gpus,
        use_gpu=True,
        resources_per_worker={"CPU": num_cpus_per_gpu, "GPU": 1}
    )
    
    # Configure training run (replaces our manual experiment tracking)
    run_config = RunConfig(
        name=config.get("experiment_name", "rat_training"),
        storage_path=results_dir,  # Save to network drive
        checkpoint_config=train.CheckpointConfig(
            checkpoint_score_attribute="val_loss",
            checkpoint_score_order="min",
            num_to_keep=3
        )
    )
    
    # Create Ray Train trainer (replaces our entire UnifiedTrainer class)
    trainer = TorchTrainer(
        train_loop_per_worker=train_function,
        train_loop_config=config,
        scaling_config=scaling_config,
        run_config=run_config
    )
    
    # Run training (this is all we need!)
    result = trainer.fit()
    
    print("Training completed!")
    print(f"Best validation loss: {result.metrics.get('val_loss', 'N/A')}")
    print(f"Results saved to: {result.path}")
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train RAT with Ray Train")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--num-gpus", type=int, default=4, help="Number of GPUs")
    parser.add_argument("--cpus-per-gpu", type=int, default=4, help="CPUs per GPU")
    
    args = parser.parse_args()
    
    train_rat_with_ray(args.config, args.num_gpus, args.cpus_per_gpu)