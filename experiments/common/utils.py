"""Common utilities for experiments."""

from datetime import datetime
import json
import logging
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
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
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
