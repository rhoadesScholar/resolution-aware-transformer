#!/usr/bin/env python3
"""
Comprehensive Evaluation Pipeline for Resolution Aware Transformer
Leverages Ray Train for distributed evaluation and provides extensive metrics
for medical segmentation, object detection, ablation studies, and robustness testing.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging
import time
import numpy as np
from dataclasses import dataclass

# Default test resolutions for robustness evaluation
DEFAULT_TEST_RESOLUTIONS = [128, 256, 512]

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
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

# Add experiments directory to path
EXPERIMENTS_DIR = Path(__file__).parent
sys.path.insert(0, str(EXPERIMENTS_DIR / "common"))

# Import common modules
from common.datasets import ISICDataset, COCODataset
from common.models import create_model
from common.metrics import SegmentationEvaluator, DetectionEvaluator
from common.utils import get_device, set_seed

# Setup logging
log_path = Path(os.environ.get("RESULTS_DIR", ".")) / "evaluation.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(str(log_path))],
)
logger = logging.getLogger(__name__)


def create_rat_model(model_config):
    """
    Factory function to create a ResolutionAwareTransformer model.
    Handles import and instantiation.
    """
    try:
        from resolution_aware_transformer import ResolutionAwareTransformer

        return ResolutionAwareTransformer(**model_config)
    except ImportError:
        logger.warning(
            "Could not import ResolutionAwareTransformer. Make sure the package is installed."
        )
        raise


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""

    task_type: str
    metrics: Dict[str, float]
    predictions: Optional[List] = None
    targets: Optional[List] = None
    metadata: Optional[Dict[str, Any]] = None


def dice_coefficient(
    pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6
) -> float:
    """Calculate Dice coefficient for segmentation."""
    pred = torch.sigmoid(pred) > 0.5
    target = target > 0.5

    intersection = (pred * target).sum()
    total = pred.sum() + target.sum()

    dice = (2.0 * intersection + smooth) / (total + smooth)
    return dice.item()


def iou_coefficient(
    pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6
) -> float:
    """Calculate IoU coefficient for segmentation."""
    pred = torch.sigmoid(pred) > 0.5
    target = target > 0.5

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou.item()


def sensitivity_specificity(
    pred: torch.Tensor, target: torch.Tensor
) -> Tuple[float, float]:
    """Calculate sensitivity and specificity for segmentation."""
    pred = torch.sigmoid(pred) > 0.5
    target = target > 0.5

    tp = (pred * target).sum().float()
    tn = ((1 - pred) * (1 - target)).sum().float()
    fp = (pred * (1 - target)).sum().float()
    fn = ((1 - pred) * target).sum().float()

    sensitivity = tp / (tp + fn + 1e-6)
    specificity = tn / (tn + fp + 1e-6)

    return sensitivity.item(), specificity.item()


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Calculate IoU between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def get_object_size_category(box: List[float]) -> str:
    """Categorize object size based on bounding box area."""
    area = (box[2] - box[0]) * (box[3] - box[1])
    if area < 32**2:
        return "small"
    elif area < 96**2:
        return "medium"
    else:
        return "large"


def calculate_ap_for_class(
    predictions: List[Dict],
    targets: List[Dict],
    class_id: int,
    iou_threshold: float = 0.5,
) -> float:
    """Calculate Average Precision for a specific class."""
    # Collect all predictions and ground truths for this class
    class_predictions = []
    class_targets = []

    for pred in predictions:
        if "boxes" in pred and "scores" in pred and "labels" in pred:
            for box, score, label in zip(pred["boxes"], pred["scores"], pred["labels"]):
                if label == class_id:
                    class_predictions.append(
                        {
                            "box": box,
                            "score": score,
                            "image_id": pred.get("image_id", 0),
                        }
                    )

    for target in targets:
        if "boxes" in target and "labels" in target:
            for box, label in zip(target["boxes"], target["labels"]):
                if label == class_id:
                    class_targets.append(
                        {
                            "box": box,
                            "image_id": target.get("image_id", 0),
                            "matched": False,
                        }
                    )

    if len(class_predictions) == 0:
        return 0.0

    # Sort predictions by confidence score
    class_predictions.sort(key=lambda x: x["score"], reverse=True)

    tp = []
    fp = []

    for pred in class_predictions:
        best_iou = 0.0
        best_target_idx = -1

        # Find best matching ground truth
        for i, target in enumerate(class_targets):
            if target["image_id"] == pred["image_id"] and not target["matched"]:
                iou = calculate_iou(pred["box"], target["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_target_idx = i

        # Check if match meets IoU threshold
        if best_iou >= iou_threshold and best_target_idx >= 0:
            tp.append(1)
            fp.append(0)
            class_targets[best_target_idx]["matched"] = True
        else:
            tp.append(0)
            fp.append(1)

    # Calculate precision and recall
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    num_gt = len(class_targets)
    if num_gt == 0:
        return 0.0

    recalls = tp_cumsum / num_gt
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)

    # Calculate AP using 11-point interpolation
    ap = 0.0
    for threshold in np.arange(0, 1.1, 0.1):
        precision_at_threshold = 0.0
        for i in range(len(recalls)):
            if recalls[i] >= threshold:
                precision_at_threshold = max(precision_at_threshold, precisions[i])
        ap += precision_at_threshold / 11.0

    return ap


def calculate_map(
    predictions: List[Dict],
    targets: List[Dict],
    iou_thresholds: Optional[List[float]] = None,
    num_classes: int = 80,  # COCO has 80 classes
) -> Dict[str, float]:
    """Calculate mAP for object detection."""
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    if len(predictions) == 0 or len(targets) == 0:
        return {
            "mAP@0.5": 0.0,
            "mAP@0.5:0.95": 0.0,
            "mAP_small": 0.0,
            "mAP_medium": 0.0,
            "mAP_large": 0.0,
        }

    # Calculate AP for each class and IoU threshold
    aps_per_class = {}
    aps_per_size = {"small": [], "medium": [], "large": []}

    # Get unique class IDs from targets
    all_class_ids = set()
    for target in targets:
        if "labels" in target:
            all_class_ids.update(target["labels"])

    for class_id in all_class_ids:
        class_aps = []

        for iou_thresh in iou_thresholds:
            ap = calculate_ap_for_class(predictions, targets, class_id, iou_thresh)
            class_aps.append(ap)

            # Collect APs by object size for IoU@0.5
            if iou_thresh == 0.5:
                # Get size category for this class's objects
                for target in targets:
                    if "boxes" in target and "labels" in target:
                        for box, label in zip(target["boxes"], target["labels"]):
                            if label == class_id:
                                size_cat = get_object_size_category(box)
                                aps_per_size[size_cat].append(ap)
                                break

        aps_per_class[class_id] = class_aps

    # Calculate overall metrics
    if len(aps_per_class) == 0:
        return {
            "mAP@0.5": 0.0,
            "mAP@0.5:0.95": 0.0,
            "mAP_small": 0.0,
            "mAP_medium": 0.0,
            "mAP_large": 0.0,
        }

    # mAP@0.5 (first threshold)
    map_50 = float(np.mean([aps[0] for aps in aps_per_class.values()]))

    # mAP@0.5:0.95 (average over all thresholds)
    map_50_95 = float(np.mean([np.mean(aps) for aps in aps_per_class.values()]))

    # mAP by object size
    map_small = float(np.mean(aps_per_size["small"])) if aps_per_size["small"] else 0.0
    map_medium = (
        float(np.mean(aps_per_size["medium"])) if aps_per_size["medium"] else 0.0
    )
    map_large = float(np.mean(aps_per_size["large"])) if aps_per_size["large"] else 0.0

    return {
        "mAP@0.5": map_50,
        "mAP@0.5:0.95": map_50_95,
        "mAP_small": map_small,
        "mAP_medium": map_medium,
        "mAP_large": map_large,
    }


def evaluate_segmentation(
    model: nn.Module, dataloader: DataLoader, device: torch.device
) -> EvaluationMetrics:
    """Evaluate segmentation model with comprehensive metrics."""
    model.eval()

    total_dice = 0.0
    total_iou = 0.0
    total_sensitivity = 0.0
    total_specificity = 0.0
    total_loss = 0.0
    num_samples = 0

    criterion = nn.BCEWithLogitsLoss()
    predictions_list = []
    targets_list = []

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, dict):
                images = batch["image"].to(device)
                targets = batch["mask"].to(device)
            else:
                images, targets = batch
                images = images.to(device)
                targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            # Calculate metrics
            batch_dice = dice_coefficient(outputs, targets)
            batch_iou = iou_coefficient(outputs, targets)
            batch_sens, batch_spec = sensitivity_specificity(outputs, targets)

            total_dice += batch_dice * images.size(0)
            total_iou += batch_iou * images.size(0)
            total_sensitivity += batch_sens * images.size(0)
            total_specificity += batch_spec * images.size(0)
            total_loss += loss.item() * images.size(0)
            num_samples += images.size(0)

            # Store predictions for detailed analysis
            predictions_list.extend(torch.sigmoid(outputs).cpu().numpy())
            targets_list.extend(targets.cpu().numpy())

    metrics = {
        "dice_coefficient": total_dice / num_samples,
        "iou_coefficient": total_iou / num_samples,
        "sensitivity": total_sensitivity / num_samples,
        "specificity": total_specificity / num_samples,
        "loss": total_loss / num_samples,
        "accuracy": (total_sensitivity + total_specificity) / (2 * num_samples),
    }

    return EvaluationMetrics(
        task_type="segmentation",
        metrics=metrics,
        predictions=predictions_list,
        targets=targets_list,
    )


def evaluate_detection(
    model: nn.Module, dataloader: DataLoader, device: torch.device
) -> EvaluationMetrics:
    """Evaluate object detection model with mAP metrics."""
    model.eval()

    predictions_list = []
    targets_list = []
    total_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if isinstance(batch, dict):
                images = batch["images"].to(device)
                batch_targets = batch.get("targets", batch.get("labels", []))
            else:
                images, batch_targets = batch
                images = images.to(device)

            # Forward pass
            outputs = model(images)

            # Handle different output formats
            if isinstance(outputs, dict):
                # Standard detection model output format
                batch_predictions = outputs
            elif isinstance(outputs, (list, tuple)):
                # Convert list/tuple outputs to dict format
                batch_predictions = {"predictions": outputs}
            else:
                # Single tensor output - need to process based on model architecture
                batch_predictions = {"logits": outputs}

            # Process each image in the batch
            batch_size = images.size(0)
            for i in range(batch_size):
                image_id = batch_idx * batch_size + i

                # Format predictions for this image
                if "boxes" in batch_predictions and "scores" in batch_predictions:
                    # Standard detection format
                    pred = {
                        "image_id": image_id,
                        "boxes": (
                            batch_predictions["boxes"][i].cpu().tolist()
                            if torch.is_tensor(batch_predictions["boxes"][i])
                            else batch_predictions["boxes"][i]
                        ),
                        "scores": (
                            batch_predictions["scores"][i].cpu().tolist()
                            if torch.is_tensor(batch_predictions["scores"][i])
                            else batch_predictions["scores"][i]
                        ),
                        "labels": (
                            batch_predictions["labels"][i].cpu().tolist()
                            if torch.is_tensor(batch_predictions["labels"][i])
                            else batch_predictions["labels"][i]
                        ),
                    }
                else:
                    # Simplified format - create dummy predictions
                    # In a real implementation, you'd extract boxes, scores, labels from model output
                    pred = {
                        "image_id": image_id,
                        "boxes": [],  # Extract from model output
                        "scores": [],  # Extract from model output
                        "labels": [],  # Extract from model output
                    }

                predictions_list.append(pred)

                # Format targets for this image
                if isinstance(batch_targets, (list, tuple)) and i < len(batch_targets):
                    target = batch_targets[i]
                    if isinstance(target, dict):
                        formatted_target = {
                            "image_id": image_id,
                            "boxes": target.get("boxes", []),
                            "labels": target.get("labels", []),
                        }
                    else:
                        # Handle tensor targets
                        formatted_target = {
                            "image_id": image_id,
                            "boxes": [],  # Extract from target tensor
                            "labels": [],  # Extract from target tensor
                        }
                elif torch.is_tensor(batch_targets) and len(batch_targets.shape) > 1:
                    # Tensor targets
                    formatted_target = {
                        "image_id": image_id,
                        "boxes": [],  # Extract from tensor
                        "labels": [],  # Extract from tensor
                    }
                else:
                    # Default empty target
                    formatted_target = {
                        "image_id": image_id,
                        "boxes": [],
                        "labels": [],
                    }

                targets_list.append(formatted_target)

            num_samples += batch_size

    # Calculate mAP metrics
    map_metrics = calculate_map(predictions_list, targets_list)

    # Calculate average loss (simplified - in practice this would be detection-specific)
    avg_loss = total_loss / max(num_samples, 1)
    metrics = {"loss": avg_loss, **map_metrics}

    return EvaluationMetrics(
        task_type="detection",
        metrics=metrics,
        predictions=predictions_list,
        targets=targets_list,
    )


def evaluate_at_multiple_resolutions(
    model: nn.Module,
    dataset_class,
    data_config: Dict[str, Any],
    device: torch.device,
    resolutions: List[int],
) -> Dict[int, EvaluationMetrics]:
    """Evaluate model at multiple resolutions for robustness testing."""
    results = {}

    for resolution in resolutions:
        logger.info(f"Evaluating at resolution {resolution}x{resolution}")

        # Create dataset with new resolution
        eval_config = data_config.copy()
        eval_config["image_size"] = resolution

        eval_dataset = dataset_class(
            data_dir=eval_config["local_data_dir"],
            split=eval_config.get("val_split", "val"),
            image_size=resolution,
            multi_scale=eval_config.get("multi_scale", False),
        )

        eval_loader = DataLoader(
            eval_dataset,
            batch_size=eval_config.get("batch_size", 8),
            shuffle=False,
            num_workers=eval_config.get("num_workers", os.cpu_count() // 2),
            pin_memory=True,
        )

        # Evaluate based on task type
        task_type = eval_config.get("task_type", "segmentation")
        if task_type == "segmentation":
            metrics = evaluate_segmentation(model, eval_loader, device)
        else:
            metrics = evaluate_detection(model, eval_loader, device)

        results[resolution] = metrics

        logger.info(f"Resolution {resolution} results: {metrics.metrics}")

    return results


def run_ablation_study(
    base_config: Dict[str, Any],
    ablation_configs: List[Dict[str, Any]],
    model_checkpoints: List[str],
    data_config: Dict[str, Any],
) -> Dict[str, EvaluationMetrics]:
    """Run ablation study evaluation on multiple model variants."""
    results = {}

    for i, (ablation_name, checkpoint_path) in enumerate(
        zip(ablation_configs, model_checkpoints)
    ):
        logger.info(f"Evaluating ablation: {ablation_name}")

        if not Path(checkpoint_path).exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            continue

        # Load model
        try:
            model_config = base_config["model"].copy()
            model_config.update(ablation_configs[i].get("model", {}))

            model = create_rat_model(model_config)

            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(checkpoint["model_state_dict"])

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)

            # Create evaluation dataset
            task_type = base_config.get("task_type", "segmentation")

            if task_type == "segmentation":
                # ISICDataset already imported from common.datasets

                eval_dataset = ISICDataset(
                    data_dir=data_config["local_data_dir"],
                    split=data_config.get("val_split", "val"),
                    image_size=data_config.get("image_size", 256),
                    multi_scale=model_config.get("multi_scale", False),
                )

                eval_loader = DataLoader(
                    eval_dataset,
                    batch_size=data_config.get("batch_size", 8),
                    shuffle=False,
                    num_workers=data_config.get("num_workers", os.cpu_count() // 2),
                    pin_memory=True,
                )

                metrics = evaluate_segmentation(model, eval_loader, device)
            else:
                # COCODataset already imported from common.datasets

                eval_dataset = COCODataset(
                    data_dir=data_config["local_data_dir"],
                    split=data_config.get("val_split", "val2017"),
                    image_size=data_config.get("image_size", 800),
                    multi_scale=model_config.get("multi_scale", False),
                )

                eval_loader = DataLoader(
                    eval_dataset,
                    batch_size=data_config.get("batch_size", 4),
                    shuffle=False,
                    num_workers=data_config.get("num_workers", os.cpu_count() // 2),
                    pin_memory=True,
                )

                metrics = evaluate_detection(model, eval_loader, device)

            results[ablation_name] = metrics
            logger.info(f"Ablation {ablation_name} results: {metrics.metrics}")

        except Exception as e:
            logger.error(f"Error evaluating ablation {ablation_name}: {e}")
            continue

    return results


def evaluation_function(config: Dict[str, Any]):
    """
    Distributed evaluation function for Ray Train integration.
    """
    # Import here to avoid Ray serialization issues
    # ISICDataset, COCODataset already imported from common.datasets
    # create_model already imported from common.models

    # Get distributed context
    rank = train.get_context().get_local_rank()
    world_size = train.get_context().get_world_size()

    logger.info(f"Worker {rank}/{world_size} starting evaluation")

    # Setup directories
    from ray_train import setup_directories

    dirs = setup_directories(config)
    local_data_dir = dirs["local_data_dir"]
    results_dir = dirs["results_dir"]

    # Load model
    model_config = config["model"].copy()
    task_type = config.get("task_type", "segmentation")

    try:
        model = create_rat_model(model_config)
    except ImportError:
        model = create_model(model_config.pop("name", "rat"), task_type, **model_config)

    # Load checkpoint
    checkpoint_path = config["evaluation"]["checkpoint_path"]
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    # Setup model for distributed evaluation
    model = train.torch.prepare_model(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create evaluation datasets
    data_config = config["data"]
    eval_config = config["evaluation"]

    if task_type == "segmentation":
        eval_dataset = ISICDataset(
            data_dir=local_data_dir,
            split=data_config.get("val_split", "val"),
            image_size=data_config.get("image_size", 256),
            multi_scale=model_config.get("multi_scale", False),
        )
    else:
        eval_dataset = COCODataset(
            data_dir=local_data_dir,
            split=data_config.get("val_split", "val2017"),
            image_size=data_config.get("image_size", 800),
            multi_scale=model_config.get("multi_scale", False),
        )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=eval_config.get("batch_size", 8),
        shuffle=False,
        num_workers=data_config.get("num_workers", os.cpu_count() // 2),
        pin_memory=True,
    )

    eval_loader = train.torch.prepare_data_loader(eval_loader)

    # Run standard evaluation
    if task_type == "segmentation":
        metrics = evaluate_segmentation(model, eval_loader, device)
    else:
        metrics = evaluate_detection(model, eval_loader, device)

    # Report metrics to Ray
    train.report(metrics.metrics)

    if rank == 0:
        logger.info(f"Evaluation completed: {metrics.metrics}")

        # Save detailed results
        results_path = Path(results_dir) / "evaluation_results.json"
        with open(results_path, "w") as f:
            json.dump(
                {
                    "task_type": task_type,
                    "metrics": metrics.metrics,
                    "config": config,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                f,
                indent=2,
            )

        logger.info(f"Results saved to {results_path}")

        # Run robustness evaluation if requested
        if eval_config.get("robustness_test", False):
            resolutions = eval_config.get("test_resolutions", DEFAULT_TEST_RESOLUTIONS)
            logger.info(f"Running robustness test at resolutions: {resolutions}")

            dataset_class = ISICDataset if task_type == "segmentation" else COCODataset
            robustness_results = evaluate_at_multiple_resolutions(
                model, dataset_class, data_config, device, resolutions
            )

            # Save robustness results
            robustness_path = Path(results_dir) / "robustness_results.json"
            robustness_data = {
                resolution: {"metrics": result.metrics, "task_type": result.task_type}
                for resolution, result in robustness_results.items()
            }

            with open(robustness_path, "w") as f:
                json.dump(robustness_data, f, indent=2)

            logger.info(f"Robustness results saved to {robustness_path}")


def evaluate_rat_with_ray(
    config_path: str,
    checkpoint_path: str,
    num_gpus: int = 1,
    robustness_test: bool = False,
    test_resolutions: Optional[List[int]] = None,
) -> None:
    """
    Evaluate RAT model using Ray Train with comprehensive metrics.

    Args:
        config_path: Path to YAML configuration file
        checkpoint_path: Path to model checkpoint
        num_gpus: Number of GPUs to use for evaluation
        robustness_test: Whether to run robustness testing
        test_resolutions: List of resolutions for robustness testing
    """
    if not RAY_AVAILABLE:
        raise ImportError("Ray is not installed. Install with: pip install ray[train]")

    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Add evaluation-specific config
    config["evaluation"] = {
        "checkpoint_path": checkpoint_path,
        "robustness_test": robustness_test,
        "test_resolutions": test_resolutions or DEFAULT_TEST_RESOLUTIONS,
        "batch_size": config.get("evaluation", {}).get("batch_size", 8),
    }

    # Initialize Ray
    if not ray.is_initialized():
        ray.init()

    logger.info(f"Evaluating {config.get('experiment_name', 'RAT experiment')}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Using {num_gpus} GPUs")
    logger.info(f"Robustness test: {robustness_test}")

    # Setup results directory
    results_config = config.get("results", {})
    results_dir = results_config.get("output_dir", "./results")
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    # Configure scaling
    scaling_config = ScalingConfig(
        num_workers=num_gpus, use_gpu=True, resources_per_worker={"CPU": 4, "GPU": 1}
    )

    # Configure evaluation run
    run_config = RunConfig(
        name=f"{config.get('experiment_name', 'rat')}_evaluation",
        storage_path=results_dir,
    )

    # Create and run trainer
    trainer = TorchTrainer(
        train_loop_per_worker=evaluation_function,
        train_loop_config=config,
        scaling_config=scaling_config,
        run_config=run_config,
    )

    result = trainer.fit()
    logger.info(f"Evaluation completed. Results: {result.metrics}")


def parse_args():
    parser = argparse.ArgumentParser(description="RAT Model Evaluation with Ray")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument(
        "--robustness", action="store_true", help="Run robustness testing"
    )
    parser.add_argument(
        "--resolutions",
        nargs="+",
        type=int,
        default=DEFAULT_TEST_RESOLUTIONS,
        help="Test resolutions for robustness",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    evaluate_rat_with_ray(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        num_gpus=args.num_gpus,
        robustness_test=args.robustness,
        test_resolutions=args.resolutions,
    )
