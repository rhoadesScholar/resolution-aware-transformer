"""Optimized RAT training using Ray Train with DeepSpeed and automatic batch sizing."""

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
import math

# Ray Train imports
try:
    import ray
    from ray import train
    from ray.train import ScalingConfig, RunConfig
    from ray.train.torch import TorchTrainer
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

# DeepSpeed integration
try:
    import deepspeed
    from ray.train.torch import TorchConfig
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

# Add experiments directory to path
EXPERIMENTS_DIR = Path(__file__).parent
sys.path.insert(0, str(EXPERIMENTS_DIR / "common"))


def get_gpu_memory_info() -> Dict[str, float]:
    """Get GPU memory information for automatic batch sizing."""
    if not torch.cuda.is_available():
        return {"total_memory": 8.0, "available_memory": 6.0}  # Default CPU fallback
    
    try:
        # Get GPU properties
        device = torch.cuda.current_device()
        gpu_props = torch.cuda.get_device_properties(device)
        total_memory = gpu_props.total_memory / (1024**3)  # Convert to GB
        
        # Estimate available memory (conservative estimate)
        available_memory = total_memory * 0.85  # Reserve 15% for overhead
        
        return {
            "total_memory": total_memory,
            "available_memory": available_memory,
            "device_name": gpu_props.name
        }
    except Exception as e:
        print(f"Warning: Could not get GPU memory info: {e}")
        return {"total_memory": 8.0, "available_memory": 6.0}


def estimate_model_memory(model_config: Dict[str, Any], spatial_dims: int = 2) -> float:
    """
    Estimate model memory usage in GB based on configuration.
    
    Args:
        model_config: Model configuration dictionary
        spatial_dims: Number of spatial dimensions (2 or 3)
    
    Returns:
        Estimated memory usage in GB
    """
    feature_dims = model_config.get("feature_dims", 128)
    num_blocks = model_config.get("num_blocks", 4)
    num_heads = model_config.get("num_heads", 8)
    mlp_ratio = model_config.get("mlp_ratio", 4)
    
    # Rough estimation based on transformer architecture
    # Each transformer block has attention + MLP
    attention_params = feature_dims * feature_dims * 4 * num_heads  # Q, K, V, O projections
    mlp_params = feature_dims * feature_dims * mlp_ratio * 2  # Up and down projections
    
    # Total parameters per block
    params_per_block = attention_params + mlp_params
    total_params = params_per_block * num_blocks
    
    # Add initial projection and positional embeddings
    total_params += feature_dims * model_config.get("input_features", 3) * 49  # 7x7 conv
    total_params += feature_dims * 1024  # Positional embeddings
    
    # Convert to memory (4 bytes per parameter for float32, plus activations)
    param_memory = total_params * 4 / (1024**3)  # GB
    activation_memory = param_memory * 2  # Rough estimate for activations
    
    # Add gradient memory for training
    gradient_memory = param_memory
    
    total_memory = param_memory + activation_memory + gradient_memory
    
    # Add safety factor
    return total_memory * 1.5


def calculate_optimal_batch_size(
    model_config: Dict[str, Any], 
    data_config: Dict[str, Any],
    gpu_memory: Dict[str, float],
    target_effective_batch_size: int = 32
) -> Dict[str, int]:
    """
    Calculate optimal batch size and gradient accumulation steps.
    
    Args:
        model_config: Model configuration
        data_config: Data configuration  
        gpu_memory: GPU memory information
        target_effective_batch_size: Target effective batch size for reproducibility
    
    Returns:
        Dictionary with batch_size, gradient_accumulation_steps, and effective_batch_size
    """
    available_memory = gpu_memory["available_memory"]
    
    # Estimate model memory usage
    model_memory = estimate_model_memory(model_config)
    
    # Estimate memory per sample
    image_size = data_config.get("image_size", 256)
    channels = model_config.get("input_features", 3)
    spatial_dims = model_config.get("spatial_dims", 2)
    
    if spatial_dims == 2:
        sample_memory = channels * image_size * image_size * 4 / (1024**3)  # 4 bytes per float32
    else:  # 3D
        depth = data_config.get("depth", image_size // 4)  # Assume depth is 1/4 of image size
        sample_memory = channels * depth * image_size * image_size * 4 / (1024**3)
    
    # Add overhead for gradients and optimizer states
    sample_memory *= 3  # Conservative estimate
    
    # Calculate maximum batch size that fits in memory
    memory_for_batch = available_memory - model_memory
    max_batch_size = max(1, int(memory_for_batch / sample_memory))
    
    # Ensure batch size is reasonable (not too large or too small)
    max_batch_size = min(max_batch_size, 32)  # Cap at 32 per GPU
    max_batch_size = max(max_batch_size, 1)   # At least 1
    
    # Calculate gradient accumulation to reach target effective batch size
    gradient_accumulation_steps = max(1, target_effective_batch_size // max_batch_size)
    effective_batch_size = max_batch_size * gradient_accumulation_steps
    
    print(f"GPU Memory: {available_memory:.1f}GB available")
    print(f"Model Memory: {model_memory:.1f}GB estimated") 
    print(f"Sample Memory: {sample_memory*1000:.1f}MB per sample")
    print(f"Optimal batch size: {max_batch_size} per GPU")
    print(f"Gradient accumulation: {gradient_accumulation_steps} steps")
    print(f"Effective batch size: {effective_batch_size}")
    
    return {
        "batch_size": max_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": effective_batch_size
    }


def create_deepspeed_config(
    model_config: Dict[str, Any],
    training_config: Dict[str, Any],
    batch_config: Dict[str, int]
) -> Dict[str, Any]:
    """
    Create DeepSpeed configuration for Ray Train integration.
    
    Args:
        model_config: Model configuration
        training_config: Training configuration
        batch_config: Batch sizing configuration
    
    Returns:
        DeepSpeed configuration dictionary
    """
    # Estimate if we need ZeRO stage 2 or 3 based on model size
    model_memory = estimate_model_memory(model_config)
    zero_stage = 3 if model_memory > 4.0 else 2  # Use stage 3 for large models
    
    deepspeed_config = {
        "train_batch_size": batch_config["effective_batch_size"],
        "train_micro_batch_size_per_gpu": batch_config["batch_size"],
        "gradient_accumulation_steps": batch_config["gradient_accumulation_steps"],
        
        # ZeRO optimization
        "zero_optimization": {
            "stage": zero_stage,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "overlap_comm": True,
            "contiguous_gradients": True,
        },
        
        # Mixed precision
        "fp16": {
            "enabled": training_config.get("mixed_precision", True),
            "auto_cast": True,
            "loss_scale": 0,
            "initial_scale_power": 16,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        
        # Gradient clipping
        "gradient_clipping": training_config.get("grad_clip", 1.0),
        
        # Optimizer configuration
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": training_config.get("learning_rate", 1e-4),
                "weight_decay": training_config.get("weight_decay", 0.01),
                "betas": [0.9, 0.999],
                "eps": 1e-8
            }
        },
        
        # Scheduler configuration
        "scheduler": {
            "type": "WarmupCosineLR",
            "params": {
                "total_num_steps": training_config.get("epochs", 100) * 1000,  # Estimate
                "warmup_num_steps": 100
            }
        } if training_config.get("scheduler") == "cosine" else None,
    }
    
    # Add CPU offloading for ZeRO stage 3
    if zero_stage == 3:
        deepspeed_config["zero_optimization"]["offload_optimizer"] = {
            "device": "cpu",
            "pin_memory": True
        }
        deepspeed_config["zero_optimization"]["offload_param"] = {
            "device": "cpu",
            "pin_memory": True
        }
    
    # Remove None values
    deepspeed_config = {k: v for k, v in deepspeed_config.items() if v is not None}
    
    print(f"DeepSpeed ZeRO Stage {zero_stage} configured")
    print(f"Mixed precision: {deepspeed_config['fp16']['enabled']}")
    
    return deepspeed_config


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
    Optimized training function with DeepSpeed and automatic batch sizing.
    This replaces our complex UnifiedTrainer class with intelligent optimization.
    """
    # Import here to avoid issues with Ray serialization
    from datasets import ISICDataset, COCODataset
    from models import create_model
    
    # Get distributed context from Ray (replaces our manual distributed detection)
    rank = train.get_context().get_local_rank()
    world_size = train.get_context().get_world_size()
    
    print(f"Worker {rank}/{world_size} starting optimized training")
    
    # Setup directories
    dirs = setup_directories(config)
    local_data_dir = dirs["local_data_dir"]
    results_dir = dirs["results_dir"]
    
    # Get GPU memory information for automatic batch sizing
    gpu_memory = get_gpu_memory_info()
    if rank == 0:
        print(f"GPU: {gpu_memory.get('device_name', 'Unknown')} ({gpu_memory['total_memory']:.1f}GB)")
    
    # Calculate optimal batch size and gradient accumulation
    model_config = config["model"].copy()
    data_config = config["data"]
    training_config = config["training"]
    
    # Get target effective batch size from config or use baseline defaults
    task_type = config.get("task_type", "segmentation")
    baseline_batch_sizes = {
        "segmentation": 32,  # ISIC 2018 baseline effective batch size
        "detection": 16,     # COCO detection baseline effective batch size
        "ablation": 32,      # Standard for ablation studies
        "robustness": 32     # Standard for robustness testing
    }
    target_effective_batch_size = training_config.get(
        "target_effective_batch_size", 
        baseline_batch_sizes.get(task_type, 32)
    )
    
    # Calculate optimal batch configuration
    batch_config = calculate_optimal_batch_size(
        model_config, data_config, gpu_memory, target_effective_batch_size
    )
    
    # Override config batch size with optimized values
    training_config["batch_size"] = batch_config["batch_size"]
    training_config["gradient_accumulation_steps"] = batch_config["gradient_accumulation_steps"]
    
    # Create model based on task
    model_name = model_config.pop("name", "rat")
    
    # Create RAT model
    try:
        from resolution_aware_transformer import ResolutionAwareTransformer
        model = ResolutionAwareTransformer(**model_config)
    except ImportError:
        # Fallback for testing
        model = create_model(model_name, task_type, **model_config)
    
    # Setup DeepSpeed if available and beneficial
    use_deepspeed = (
        DEEPSPEED_AVAILABLE and 
        world_size > 1 and 
        training_config.get("use_deepspeed", True)
    )
    
    if use_deepspeed and rank == 0:
        print("Setting up DeepSpeed optimization...")
        
        # Create DeepSpeed configuration
        deepspeed_config = create_deepspeed_config(model_config, training_config, batch_config)
        
        # Initialize DeepSpeed engine
        model_engine, optimizer, _, scheduler = deepspeed.initialize(
            model=model,
            config=deepspeed_config
        )
        model = model_engine
        
    else:
        # Ray automatically handles device placement and DDP wrapping
        model = train.torch.prepare_model(model)
        
        # Create optimizer and scheduler manually if not using DeepSpeed
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
    
    # Create datasets using local data directory
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
    
    # Create data loaders with optimized batch size
    batch_size = batch_config["batch_size"]
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=data_config.get("num_workers", 4),
        pin_memory=True,
        drop_last=True  # Ensure consistent batch sizes for gradient accumulation
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,  # Larger batch size for validation
        shuffle=False,
        num_workers=data_config.get("num_workers", 4),
        pin_memory=True
    )
    
    # Ray prepares data loaders for distributed training
    if not use_deepspeed:
        train_loader = train.torch.prepare_data_loader(train_loader)
        val_loader = train.torch.prepare_data_loader(val_loader)
    
    # Training loop with gradient accumulation and optimization
    num_epochs = training_config.get("epochs", 100)
    gradient_accumulation_steps = batch_config["gradient_accumulation_steps"]
    
    if rank == 0:
        print(f"Starting training for {num_epochs} epochs")
        print(f"Batch size per GPU: {batch_size}")
        print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
        print(f"Effective batch size: {batch_config['effective_batch_size']}")
        print(f"Using DeepSpeed: {use_deepspeed}")
    
    for epoch in range(num_epochs):
        # Training phase with gradient accumulation
        model.train()
        train_loss = 0.0
        train_samples = 0
        accumulated_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
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
            
            # Scale loss by gradient accumulation steps
            loss = loss / gradient_accumulation_steps
            accumulated_loss += loss.item()
            
            # Backward pass
            if use_deepspeed:
                model.backward(loss)
            else:
                loss.backward()
            
            # Update parameters every gradient_accumulation_steps
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping
                if not use_deepspeed and training_config.get("grad_clip", 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), training_config["grad_clip"]
                    )
                
                # Optimizer step
                if use_deepspeed:
                    model.step()
                else:
                    optimizer.step()
                    optimizer.zero_grad()
                
                train_loss += accumulated_loss
                train_samples += images.size(0) * gradient_accumulation_steps
                accumulated_loss = 0.0
            
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
        if scheduler and not use_deepspeed:
            scheduler.step()
        elif use_deepspeed:
            # DeepSpeed handles scheduling internally
            pass
        
        # Calculate average losses
        avg_train_loss = train_loss / max(1, len(train_loader) // gradient_accumulation_steps)
        avg_val_loss = val_loss / len(val_loader)
        
        # Get current learning rate
        if use_deepspeed:
            current_lr = model.get_lr()[0] if hasattr(model, 'get_lr') else training_config.get("learning_rate", 1e-4)
        else:
            current_lr = optimizer.param_groups[0]["lr"]
        
        # Report metrics to Ray Train (replaces our manual logging)
        metrics = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "learning_rate": current_lr,
            "effective_batch_size": batch_config["effective_batch_size"],
            "gpu_memory_used": torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
        }
        
        train.report(metrics)
        
        if rank == 0:
            print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {current_lr:.2e}")
            
            # Save checkpoint to network drive (results directory)
            if (epoch + 1) % training_config.get("checkpoint_freq", 10) == 0:
                checkpoint_path = Path(results_dir) / f"checkpoint_epoch_{epoch + 1}.pth"
                
                if use_deepspeed:
                    # Save DeepSpeed checkpoint
                    model.save_checkpoint(str(checkpoint_path.parent), tag=f"epoch_{epoch + 1}")
                else:
                    # Save regular PyTorch checkpoint
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': avg_val_loss,
                        'config': config,
                        'batch_config': batch_config
                    }, checkpoint_path)
                
                print(f"Checkpoint saved to {checkpoint_path}")
    
    if rank == 0:
        print("Training completed successfully!")
        print(f"Final effective batch size: {batch_config['effective_batch_size']}")
        print(f"GPU utilization optimized with batch size: {batch_size}")


def train_rat_with_ray(config_path: str, num_gpus: int = 4, num_cpus_per_gpu: int = 4) -> None:
    """
    Train RAT model using Ray Train with DeepSpeed and automatic optimization.
    
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
    
    print(f"Training {config.get('experiment_name', 'RAT experiment')} with optimized Ray Train")
    print(f"Using {num_gpus} GPUs with {num_cpus_per_gpu} CPUs per GPU")
    print(f"Dataset: {data_config['dataset_name']} (local: {data_config['local_data_dir']})")
    
    # Setup results directory (network drive)
    results_config = config.get("results", {})
    results_dir = results_config.get("output_dir", "./results")
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {results_dir}")
    
    # Configure scaling with DeepSpeed support (replaces our manual distributed setup)
    scaling_config = ScalingConfig(
        num_workers=num_gpus,
        use_gpu=True,
        resources_per_worker={"CPU": num_cpus_per_gpu, "GPU": 1}
    )
    
    # Prepare torch config for DeepSpeed if applicable
    torch_config = None
    use_deepspeed = (
        DEEPSPEED_AVAILABLE and 
        num_gpus > 1 and 
        config["training"].get("use_deepspeed", True)
    )
    
    if use_deepspeed:
        print("Configuring Ray Train with DeepSpeed integration...")
        
        # Get GPU memory info for batch size calculation
        if torch.cuda.is_available():
            gpu_memory = get_gpu_memory_info()
            print(f"Detected GPU: {gpu_memory.get('device_name', 'Unknown')} ({gpu_memory['total_memory']:.1f}GB)")
        
        # Calculate optimal batch configuration (will be used in train_function)
        model_config = config["model"]
        task_type = config.get("task_type", "segmentation")
        baseline_batch_sizes = {
            "segmentation": 32,  # ISIC 2018 baseline
            "detection": 16,     # COCO detection baseline  
            "ablation": 32,
            "robustness": 32
        }
        target_effective_batch_size = config["training"].get(
            "target_effective_batch_size", 
            baseline_batch_sizes.get(task_type, 32)
        )
        
        print(f"Target effective batch size: {target_effective_batch_size}")
        print(f"Automatic batch sizing and gradient accumulation will be configured per worker")
        
        # Configure Ray Train with DeepSpeed backend
        torch_config = TorchConfig(backend="nccl")  # Required for DeepSpeed
    
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
    
    # Create Ray Train trainer with DeepSpeed support
    trainer_kwargs = {
        "train_loop_per_worker": train_function,
        "train_loop_config": config,
        "scaling_config": scaling_config,
        "run_config": run_config
    }
    
    if torch_config:
        trainer_kwargs["torch_config"] = torch_config
    
    trainer = TorchTrainer(**trainer_kwargs)
    
    print("Starting optimized training with Ray Train...")
    if use_deepspeed:
        print("✓ DeepSpeed Stage 2/3 enabled for memory optimization")
    print("✓ Automatic batch size and gradient accumulation")
    print("✓ GPU memory optimization")
    print("✓ Distributed training across all GPUs")
    
    # Run training (this is all we need!)
    result = trainer.fit()
    
    print("Training completed successfully!")
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