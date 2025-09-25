"""Common utilities for experiments."""

from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)
from pathlib import Path
import random
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_dir: str, experiment_name: str) -> logging.Logger:
    """Setup logging for experiments."""
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir_path / f"{experiment_name}_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    logger = logging.getLogger(experiment_name)
    return logger


def save_config(config: Dict[str, Any], save_path: str):
    """Save experiment configuration."""
    save_path_obj = Path(save_path)
    save_path_obj.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path_obj, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def get_optimal_batch_size(
    model, sample_input, device, max_batch_size=128, start_batch_size=None
):
    """
    Automatically find the optimal batch size for the given model and GPU memory.

    Args:
        model: The PyTorch model
        sample_input: A sample input tensor (batch_size=1)
        device: The device to run on
        max_batch_size: Maximum batch size to try
        start_batch_size: Starting batch size (if None, starts from config)

    Returns:
        Optimal batch size that fits in memory
    """
    import gc

    model.eval()

    if start_batch_size is None:
        batch_size = 2
    else:
        batch_size = start_batch_size

    optimal_batch_size = 1

    device_name = (
        torch.cuda.get_device_name(device) if torch.cuda.is_available() else str(device)
    )
    logger.info(f"Finding optimal batch size for device: {device_name}...")

    while batch_size <= max_batch_size:
        try:
            # Clear cache
            torch.cuda.empty_cache()
            gc.collect()

            # Create batch
            if isinstance(sample_input, dict):
                batch = {
                    k: v.repeat(batch_size, *([1] * (v.dim() - 1)))
                    for k, v in sample_input.items()
                }
            else:
                batch = sample_input.repeat(
                    batch_size, *([1] * (sample_input.dim() - 1))
                )

            # Test forward pass
            with torch.no_grad():
                if isinstance(batch, dict):
                    _ = model(**batch)
                else:
                    _ = model(batch)

            optimal_batch_size = batch_size
            memory_used = torch.cuda.max_memory_allocated(device) / 1024**3  # GB
            logger.info(f"Batch size {batch_size}: OK (Memory: {memory_used:.1f}GB)")

            batch_size *= 2

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.info(f"Batch size {batch_size}: OOM")
                break
            else:
                raise e

    logger.info(f"Optimal batch size: {optimal_batch_size}")
    model.train()
    return optimal_batch_size


def adjust_config_for_gpu_memory(
    config: Dict[str, Any], gpu_memory_gb: int = None
) -> Dict[str, Any]:
    """
    Dynamically adjust configuration based on available GPU memory.

    Args:
        config: Original configuration dictionary
        gpu_memory_gb: GPU memory in GB (if None, tries to detect from config or defaults to 80GB)

    Returns:
        Memory-optimized configuration
    """
    config = config.copy()

    # Auto-detect GPU memory from cluster config if not provided
    if gpu_memory_gb is None:
        # Try to detect GPU memory using PyTorch
        try:
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                props = torch.cuda.get_device_properties(device)
                gpu_memory_gb = props.total_memory // (1024 ** 3)
            else:
                gpu_memory_gb = None
        except Exception as e:
            logger.warning(f"Could not detect GPU memory via PyTorch: {e}")
            gpu_memory_gb = None
        # If still not detected, try config file
        if gpu_memory_gb is None:
            try:
                import configparser
                from pathlib import Path
                config_file = Path(__file__).parent.parent / ".config"
                if config_file.exists():
                    cluster_config = configparser.ConfigParser()
                    cluster_config.read(config_file)
                    memory_mb = int(
                        cluster_config.get("cluster", "memory_mb_per_gpu", fallback="80000")
                    )
                    gpu_memory_gb = memory_mb // 1000
                else:
                    gpu_memory_gb = 80  # Default to H100
            except Exception as e:
                logger.warning(f"Could not detect GPU memory via config file: {e}")
                gpu_memory_gb = 80  # Fallback to H100

    logger.info(f"Optimizing for GPU with {gpu_memory_gb}GB memory...")

    # Get image size and task type
    if "data" in config:
        image_size = config["data"].get("image_size", 256)
        is_detection = "coco" in config["data"].get("dataset", "").lower()
    else:
        image_size = 256
        is_detection = False

    # Get model complexity
    is_multiscale = config.get("model", {}).get("multi_scale", False)
    feature_dims = config.get("model", {}).get("feature_dims", 256)

    # Memory scaling factor based on available memory
    # Base calculations are for 80GB, scale accordingly
    memory_scale = gpu_memory_gb / 80.0

    # Calculate optimal batch sizes based on memory and task
    if is_detection:
        # Object detection (larger images, more memory per sample)
        if is_multiscale:
            base_batch = max(2, int(4 * memory_scale))  # Conservative for multi-scale
        else:
            base_batch = max(
                4, int(8 * memory_scale)
            )  # More aggressive for single-scale

        # Scale down for very large images
        if image_size > 800:
            base_batch = max(1, base_batch // 2)

    else:
        # Segmentation (smaller images, less memory per sample)
        if is_multiscale:
            base_batch = max(8, int(16 * memory_scale))  # Conservative for multi-scale
        else:
            base_batch = max(
                16, int(32 * memory_scale)
            )  # More aggressive for single-scale

        # Scale up for smaller images
        if image_size <= 256:
            base_batch = int(base_batch * 1.5)

    # Adjust for model complexity
    if feature_dims > 256:
        base_batch = max(1, base_batch // 2)

    # Cap batch sizes to reasonable maximums
    max_batch = 128 if not is_detection else 32
    optimal_batch = min(base_batch, max_batch)

    # Update batch sizes in config
    if "training" in config:
        old_batch = config["training"].get("batch_size", 16)
        config["training"]["batch_size"] = optimal_batch
        logger.info(
            f"Training batch size: {old_batch} → {optimal_batch} ({gpu_memory_gb}GB GPU)"
        )

    if "data" in config:
        old_batch = config["data"].get("batch_size", 16)
        config["data"]["batch_size"] = optimal_batch
        logger.info(
            f"Data batch size: {old_batch} → {optimal_batch} ({gpu_memory_gb}GB GPU)"
        )

    if "evaluation" in config:
        old_batch = config["evaluation"].get("batch_size", 16)
        eval_batch = min(optimal_batch * 2, max_batch)  # Can use larger batch for eval
        config["evaluation"]["batch_size"] = eval_batch
        logger.info(
            f"Eval batch size: {old_batch} → {eval_batch} ({gpu_memory_gb}GB GPU)"
        )

    # Optimize number of workers based on batch size and memory
    optimal_workers = min(20, max(4, optimal_batch // 2))
    if "data" in config:
        old_workers = config["data"].get("num_workers", 4)
        config["data"]["num_workers"] = optimal_workers
        logger.info(f"Data workers: {old_workers} → {optimal_workers}")

    return config


# Keep the old function for backward compatibility but mark as deprecated
def adjust_config_for_h200(config: Dict[str, Any]) -> Dict[str, Any]:
    """Deprecated: Use adjust_config_for_gpu_memory() instead."""
    return adjust_config_for_gpu_memory(config, gpu_memory_gb=140)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load experiment configuration."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def save_results(results: Dict[str, Any], save_path: str):
    """Save experiment results."""
    save_path_obj = Path(save_path)
    save_path_obj.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path_obj, "w") as f:
        json.dump(results, f, indent=2)


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Get the best available device."""
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    return device


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """Format time in seconds to human readable format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, name: str, fmt: str = ":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ExperimentTracker:
    """Track experiment metrics and save results."""

    def __init__(self, experiment_name: str, save_dir: str):
        self.experiment_name = experiment_name
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.metrics = {}
        self.config = {}
        self.start_time = None

    def log_config(self, config: Dict[str, Any]):
        """Log experiment configuration."""
        self.config = config
        save_config(config, str(self.save_dir / "config.yaml"))

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Log a metric value."""
        if key not in self.metrics:
            self.metrics[key] = []

        entry = {"value": value}
        if step is not None:
            entry["step"] = step

        self.metrics[key].append(entry)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log multiple metrics at once."""
        for key, value in metrics.items():
            self.log_metric(key, value, step)

    def start_timer(self):
        """Start timing the experiment."""
        self.start_time = datetime.now()

    def end_timer(self):
        """End timing and log duration."""
        if self.start_time is not None:
            duration = (datetime.now() - self.start_time).total_seconds()
            self.log_metric("experiment_duration_seconds", duration)
            return duration
        return None

    def save_results(self):
        """Save all tracked results."""
        results = {
            "experiment_name": self.experiment_name,
            "config": self.config,
            "metrics": self.metrics,
            "timestamp": datetime.now().isoformat(),
        }

        if self.start_time is not None:
            results["duration_seconds"] = (
                datetime.now() - self.start_time
            ).total_seconds()

        save_results(results, str(self.save_dir / "results.json"))
        return results


def create_experiment_dir(base_dir: str, experiment_name: str) -> Path:
    """Create a timestamped experiment directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(base_dir) / f"{experiment_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


# Baseline performance references for comparison
BASELINE_SCORES = {
    "isic2018": {
        "unet": {"dice": 0.847, "iou": 0.765},
        "swin_unet": {"dice": 0.863, "iou": 0.781},
        "transunet": {"dice": 0.855, "iou": 0.773},
    },
    "coco2017": {
        "detr": {"mAP": 42.0, "mAP_small": 20.5, "mAP_medium": 45.8, "mAP_large": 61.1},
        "yolov8": {
            "mAP": 50.2,
            "mAP_small": 31.8,
            "mAP_medium": 53.9,
            "mAP_large": 70.1,
        },
    },
}


def get_baseline_score(dataset: str, model: str, metric: str) -> Optional[float]:
    """Get baseline score for comparison."""
    try:
        return BASELINE_SCORES[dataset][model][metric]
    except KeyError:
        return None


def calculate_improvement(our_score: float, baseline_score: float) -> float:
    """Calculate percentage improvement over baseline."""
    return ((our_score - baseline_score) / baseline_score) * 100
