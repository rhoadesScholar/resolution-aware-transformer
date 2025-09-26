"""Evaluation metrics for experiments using torchmetrics."""

from typing import Dict, List, Tuple, Optional
import time
import warnings

import numpy as np
import torch

# Import torchmetrics - will fail if not installed
from torchmetrics.detection import MeanAveragePrecision
from torchmetrics.segmentation import DiceScore, MeanIoU
from torchmetrics.classification import BinarySpecificity, BinaryRecall


def dice_coefficient(
    pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6
) -> torch.Tensor:
    """
    Calculate Dice coefficient for binary segmentation using torchmetrics.

    Args:
        pred: Predicted segmentation [B, H, W] or [B, 1, H, W]
        target: Ground truth segmentation [B, H, W] or [B, 1, H, W]
        smooth: Smoothing factor (unused, kept for API compatibility)

    Returns:
        Dice coefficient
    """
    # Use torchmetrics implementation (suppress deprecation warning)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Argument `average`.*is deprecated.*",
            category=DeprecationWarning,
        )
        dice_metric = DiceScore(
            num_classes=2,
            include_background=False,
            input_format="index",
            average="micro",
        )

    # Process predictions and targets for index format
    if pred.dim() == 4 and pred.size(1) == 1:
        pred = pred.squeeze(1)
    if target.dim() == 4 and target.size(1) == 1:
        target = target.squeeze(1)

    # Apply sigmoid and convert to binary classes
    if pred.max() > 1 or pred.min() < 0:
        pred = torch.sigmoid(pred)

    pred_binary = (pred > 0.5).long()
    target_binary = (target > 0.5).long()

    return dice_metric(pred_binary, target_binary)


def iou_score(
    pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6
) -> torch.Tensor:
    """
    Calculate Intersection over Union (IoU) for binary segmentation using torchmetrics.

    Args:
        pred: Predicted segmentation [B, H, W] or [B, 1, H, W]
        target: Ground truth segmentation [B, H, W] or [B, 1, H, W]
        smooth: Smoothing factor (unused, kept for API compatibility)

    Returns:
        IoU score
    """
    # Use torchmetrics implementation
    iou_metric = MeanIoU(num_classes=2)  # Binary segmentation: background + foreground

    # Ensure proper shape and format
    if pred.dim() == 4 and pred.size(1) == 1:
        pred = pred.squeeze(1)
    if target.dim() == 4 and target.size(1) == 1:
        target = target.squeeze(1)

    # Apply sigmoid and convert to class predictions
    if pred.max() > 1 or pred.min() < 0:
        pred = torch.sigmoid(pred)
    pred_classes = (pred > 0.5).long()
    target_classes = (target > 0.5).long()

    return iou_metric(pred_classes, target_classes)


def sensitivity_specificity(
    pred: torch.Tensor, target: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate sensitivity (recall) and specificity using torchmetrics.

    Args:
        pred: Predicted segmentation [B, H, W] or [B, 1, H, W]
        target: Ground truth segmentation [B, H, W] or [B, 1, H, W]

    Returns:
        Tuple of (sensitivity, specificity)
    """
    # Use torchmetrics implementations
    recall_metric = BinaryRecall()
    specificity_metric = BinarySpecificity()

    # Process predictions and targets
    if pred.dim() == 4 and pred.size(1) == 1:
        pred = pred.squeeze(1)
    if target.dim() == 4 and target.size(1) == 1:
        target = target.squeeze(1)

    # Apply sigmoid if needed
    if pred.max() > 1 or pred.min() < 0:
        pred = torch.sigmoid(pred)

    # Convert to binary
    target_binary = (target > 0.5).int()

    sensitivity = recall_metric(pred, target_binary)
    specificity = specificity_metric(pred, target_binary)

    return sensitivity, specificity


def segmentation_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """
    Calculate comprehensive segmentation metrics.

    Args:
        pred: Predicted segmentation
        target: Ground truth segmentation

    Returns:
        Dictionary of metrics
    """
    dice = dice_coefficient(pred, target)
    iou = iou_score(pred, target)
    sensitivity, specificity = sensitivity_specificity(pred, target)

    return {
        "dice": dice.item(),
        "iou": iou.item(),
        "sensitivity": sensitivity.item(),
        "specificity": specificity.item(),
    }


def mean_average_precision(
    pred_boxes: List[torch.Tensor],
    pred_scores: List[torch.Tensor],
    pred_labels: List[torch.Tensor],
    target_boxes: List[torch.Tensor],
    target_labels: List[torch.Tensor],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Calculate mean Average Precision (mAP) for object detection using torchmetrics.

    Args:
        pred_boxes: List of predicted bounding boxes per image [N, 4]
        pred_scores: List of prediction scores per image [N]
        pred_labels: List of predicted labels per image [N]
        target_boxes: List of ground truth boxes per image [M, 4]
        target_labels: List of ground truth labels per image [M]
        iou_threshold: IoU threshold (unused, kept for API compatibility)

    Returns:
        Dictionary with mAP metrics
    """
    # Initialize torchmetrics mAP calculator
    map_metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        iou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
        rec_thresholds=None,
        max_detection_thresholds=[1, 10, 100],
        class_metrics=False,
        average="macro",
    )

    # Convert data to torchmetrics format
    preds = []
    targets = []

    for i in range(len(pred_boxes)):
        # Predictions for this image
        if len(pred_boxes[i]) > 0:
            preds.append(
                {
                    "boxes": pred_boxes[i],
                    "scores": pred_scores[i],
                    "labels": pred_labels[i].long(),
                }
            )
        else:
            # Empty prediction
            preds.append(
                {
                    "boxes": torch.empty((0, 4)),
                    "scores": torch.empty((0,)),
                    "labels": torch.empty((0,), dtype=torch.long),
                }
            )

    for i in range(len(target_boxes)):
        # Targets for this image
        if len(target_boxes[i]) > 0:
            targets.append(
                {
                    "boxes": target_boxes[i],
                    "labels": target_labels[i].long(),
                }
            )
        else:
            # Empty target
            targets.append(
                {
                    "boxes": torch.empty((0, 4)),
                    "labels": torch.empty((0,), dtype=torch.long),
                }
            )

    # Handle empty predictions/targets
    if len(preds) == 0 or len(targets) == 0:
        return {
            "mAP": 0.0,
            "mAP_50": 0.0,
            "mAP_75": 0.0,
            "mAP_small": 0.0,
            "mAP_medium": 0.0,
            "mAP_large": 0.0,
        }

    # Update metric with predictions and targets
    map_metric.update(preds, targets)

    # Compute results
    results = map_metric.compute()

    # Convert to float for JSON serialization
    return {
        "mAP": float(results.get("map", 0.0)),
        "mAP_50": float(results.get("map_50", 0.0)),
        "mAP_75": float(results.get("map_75", 0.0)),
        "mAP_small": float(results.get("map_small", 0.0)),
        "mAP_medium": float(results.get("map_medium", 0.0)),
        "mAP_large": float(results.get("map_large", 0.0)),
    }


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Calculate IoU between two sets of boxes.

    Args:
        boxes1: [N, 4] in format (x1, y1, x2, y2)
        boxes2: [M, 4] in format (x1, y1, x2, y2)

    Returns:
        IoU matrix [N, M]
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    union = area1[:, None] + area2 - inter


class SegmentationEvaluator:
    """Evaluator for segmentation tasks using torchmetrics."""

    def __init__(self):
        """Initialize torchmetrics-based evaluator."""
        # Suppress deprecation warning for DiceScore
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*average.*")
            self.dice_metric = DiceScore(
                num_classes=2,
                include_background=False,
                input_format="index",
                average="micro",
            )
        self.iou_metric = MeanIoU(num_classes=2)  # Binary: background + foreground
        self.sensitivity_metric = BinaryRecall()
        self.specificity_metric = BinarySpecificity()

    def reset(self):
        """Reset accumulated metrics."""
        self.dice_metric.reset()
        self.iou_metric.reset()
        self.sensitivity_metric.reset()
        self.specificity_metric.reset()

    @staticmethod
    def _preprocess(
        pred: torch.Tensor, target: torch.Tensor, binary_type: str = "long"
    ):
        """
        Preprocess prediction and target tensors for metric computation.

        Args:
            pred: Predicted tensor
            target: Ground truth tensor
            binary_type: "long" for Dice/IoU, "int" for sensitivity/specificity

        Returns:
            Tuple of (processed_pred, processed_target)
        """
        if pred.dim() == 4 and pred.size(1) == 1:
            pred = pred.squeeze(1)
        if target.dim() == 4 and target.size(1) == 1:
            target = target.squeeze(1)

        if pred.max() > 1 or pred.min() < 0:
            pred = torch.sigmoid(pred)

        if binary_type == "long":
            pred_bin = (pred > 0.5).long()
            target_bin = (target > 0.5).long()
        else:
            pred_bin = pred
            target_bin = (target > 0.5).int()

        return pred_bin, target_bin

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """Update metrics with new batch."""
        # Dice
        dice_pred, dice_target = self._preprocess(pred, target, binary_type="long")
        self.dice_metric.update(dice_pred, dice_target)

        # IoU
        iou_pred, iou_target = self._preprocess(pred, target, binary_type="long")
        self.iou_metric.update(iou_pred, iou_target)

        # Sensitivity/Specificity
        sens_pred, sens_target = self._preprocess(pred, target, binary_type="int")
        self.sensitivity_metric.update(sens_pred, sens_target)
        self.specificity_metric.update(sens_pred, sens_target)

    def compute(self) -> Dict[str, float]:
        """Compute final metrics."""
        return {
            "dice": float(self.dice_metric.compute()),
            "iou": float(self.iou_metric.compute()),
            "sensitivity": float(self.sensitivity_metric.compute()),
            "specificity": float(self.specificity_metric.compute()),
        }


class DetectionEvaluator:
    """Evaluator for object detection tasks using torchmetrics."""

    def __init__(self):
        """Initialize torchmetrics-based evaluator."""
        self.map_metric = MeanAveragePrecision(
            box_format="xyxy",
            iou_type="bbox",
            iou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
            rec_thresholds=None,
            max_detection_thresholds=[1, 10, 100],
            class_metrics=False,
            average="macro",
        )

    def reset(self):
        """Reset accumulated predictions."""
        self.map_metric.reset()

    def update(
        self,
        pred_boxes: torch.Tensor,
        pred_scores: torch.Tensor,
        pred_labels: torch.Tensor,
        target_boxes: torch.Tensor,
        target_labels: torch.Tensor,
    ):
        """Update with new batch."""
        # Convert single tensors to lists for torchmetrics format
        preds = [
            {
                "boxes": pred_boxes,
                "scores": pred_scores,
                "labels": pred_labels.long(),
            }
        ]
        targets = [
            {
                "boxes": target_boxes,
                "labels": target_labels.long(),
            }
        ]

        self.map_metric.update(preds, targets)

    def compute(self) -> Dict[str, float]:
        """Compute final metrics."""
        results = self.map_metric.compute()

        return {
            "mAP": float(results.get("map", 0.0)),
            "mAP_50": float(results.get("map_50", 0.0)),
            "mAP_75": float(results.get("map_75", 0.0)),
            "mAP_small": float(results.get("map_small", 0.0)),
            "mAP_medium": float(results.get("map_medium", 0.0)),
            "mAP_large": float(results.get("map_large", 0.0)),
        }


def calculate_flops(model: torch.nn.Module, input_shape: Tuple[int, ...]) -> int:
    """
    Calculate FLOPs for a model (simplified estimation).

    For accurate FLOP counting, use libraries like fvcore or ptflops.
    """
    # This is a placeholder - implement actual FLOP counting
    # You can use libraries like:
    # - fvcore.nn.FlopCountMode
    # - ptflops
    # - torchprofile

    total_params = sum(p.numel() for p in model.parameters())
    # Rough estimation: assume 2 FLOPs per parameter per forward pass
    estimated_flops = total_params * 2

    return estimated_flops


def benchmark_model(
    model: torch.nn.Module, input_tensor: torch.Tensor, num_runs: int = 100
) -> Dict[str, float]:
    """
    Benchmark model inference time and memory usage.

    Args:
        model: The model to benchmark
        input_tensor: Input tensor for inference
        num_runs: Number of runs for averaging

    Returns:
        Dictionary with timing and memory statistics
    """
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Timing
    times = []

    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()

            start_time = time.time()
            _ = model(input_tensor)

            if device.type == "cuda":
                torch.cuda.synchronize()

            end_time = time.time()
            times.append(end_time - start_time)

    # Memory usage (if CUDA)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        with torch.no_grad():
            _ = model(input_tensor)

        peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
    else:
        peak_memory = 0.0

    return {
        "avg_inference_time": float(np.mean(times)),
        "std_inference_time": float(np.std(times)),
        "min_inference_time": float(np.min(times)),
        "max_inference_time": float(np.max(times)),
        "peak_memory_mb": float(peak_memory),
        "throughput_fps": float(1.0 / np.mean(times)),
    }
