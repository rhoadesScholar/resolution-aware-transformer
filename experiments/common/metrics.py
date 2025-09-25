"""Evaluation metrics for experiments."""

from typing import Dict, List, Tuple
import warnings
import time

import numpy as np
import torch


def dice_coefficient(
    pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6
) -> torch.Tensor:
    """
    Calculate Dice coefficient for binary segmentation.

    Args:
        pred: Predicted segmentation [B, H, W] or [B, 1, H, W]
        target: Ground truth segmentation [B, H, W] or [B, 1, H, W]
        smooth: Smoothing factor to avoid division by zero

    Returns:
        Dice coefficient
    """
    # Ensure same shape
    if pred.dim() == 4 and pred.size(1) == 1:
        pred = pred.squeeze(1)
    if target.dim() == 4 and target.size(1) == 1:
        target = target.squeeze(1)

    # Flatten
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)

    # Apply sigmoid if needed (assume logits)
    if pred.max() > 1 or pred.min() < 0:
        pred_flat = torch.sigmoid(pred_flat)

    # Ensure target is binary
    target_flat = (target_flat > 0.5).float()

    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.mean()


def iou_score(
    pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6
) -> torch.Tensor:
    """
    Calculate Intersection over Union (IoU) for binary segmentation.

    Args:
        pred: Predicted segmentation [B, H, W] or [B, 1, H, W]
        target: Ground truth segmentation [B, H, W] or [B, 1, H, W]
        smooth: Smoothing factor to avoid division by zero

    Returns:
        IoU score
    """
    # Ensure same shape
    if pred.dim() == 4 and pred.size(1) == 1:
        pred = pred.squeeze(1)
    if target.dim() == 4 and target.size(1) == 1:
        target = target.squeeze(1)

    # Flatten
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)

    # Apply sigmoid if needed (assume logits)
    if pred.max() > 1 or pred.min() < 0:
        pred_flat = torch.sigmoid(pred_flat)

    # Ensure target is binary
    target_flat = (target_flat > 0.5).float()

    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()


def sensitivity_specificity(
    pred: torch.Tensor, target: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate sensitivity (recall) and specificity.

    Args:
        pred: Predicted segmentation [B, H, W] or [B, 1, H, W]
        target: Ground truth segmentation [B, H, W] or [B, 1, H, W]

    Returns:
        Tuple of (sensitivity, specificity)
    """
    # Ensure same shape
    if pred.dim() == 4 and pred.size(1) == 1:
        pred = pred.squeeze(1)
    if target.dim() == 4 and target.size(1) == 1:
        target = target.squeeze(1)

    # Flatten and convert to binary
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    # Apply sigmoid if needed
    if pred.max() > 1 or pred.min() < 0:
        pred_flat = torch.sigmoid(pred_flat)

    pred_binary = (pred_flat > 0.5).float()
    target_binary = (target_flat > 0.5).float()

    # Calculate confusion matrix components
    tp = (pred_binary * target_binary).sum()
    tn = ((1 - pred_binary) * (1 - target_binary)).sum()
    fp = (pred_binary * (1 - target_binary)).sum()
    fn = ((1 - pred_binary) * target_binary).sum()

    sensitivity = tp / (tp + fn + 1e-6)  # True Positive Rate
    specificity = tn / (tn + fp + 1e-6)  # True Negative Rate

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
    Calculate mean Average Precision (mAP) for object detection.

    This is a simplified implementation. For production use, consider
    torchmetrics or pycocotools.

    Args:
        pred_boxes: List of predicted bounding boxes per image [N, 4]
        pred_scores: List of prediction scores per image [N]
        pred_labels: List of predicted labels per image [N]
        target_boxes: List of ground truth boxes per image [M, 4]
        target_labels: List of ground truth labels per image [M]
        iou_threshold: IoU threshold for matching

    Returns:
        Dictionary with mAP metrics
    """
    # This is a placeholder implementation
    # In practice, you would use torchmetrics.MeanAveragePrecision or pycocotools
    warnings.warn(
        "This is a simplified mAP implementation. Use torchmetrics for production."
    )

    # For now, return dummy values - replace with actual implementation
    return {
        "mAP": 0.0,
        "mAP_50": 0.0,
        "mAP_75": 0.0,
        "mAP_small": 0.0,
        "mAP_medium": 0.0,
        "mAP_large": 0.0,
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
    iou = inter / union
    return iou


class SegmentationEvaluator:
    """Evaluator for segmentation tasks."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset accumulated metrics."""
        self.dice_scores = []
        self.iou_scores = []
        self.sensitivity_scores = []
        self.specificity_scores = []

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """Update with new batch."""
        metrics = segmentation_metrics(pred, target)
        self.dice_scores.append(metrics["dice"])
        self.iou_scores.append(metrics["iou"])
        self.sensitivity_scores.append(metrics["sensitivity"])
        self.specificity_scores.append(metrics["specificity"])

    def compute(self) -> Dict[str, float]:
        """Compute final metrics."""
        if not self.dice_scores:
            return {"dice": 0.0, "iou": 0.0, "sensitivity": 0.0, "specificity": 0.0}

        return {
            "dice": np.mean(self.dice_scores),
            "iou": np.mean(self.iou_scores),
            "sensitivity": np.mean(self.sensitivity_scores),
            "specificity": np.mean(self.specificity_scores),
            "dice_std": np.std(self.dice_scores),
            "iou_std": np.std(self.iou_scores),
        }


class DetectionEvaluator:
    """Evaluator for object detection tasks."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset accumulated predictions."""
        self.predictions = []
        self.targets = []

    def update(
        self,
        pred_boxes: torch.Tensor,
        pred_scores: torch.Tensor,
        pred_labels: torch.Tensor,
        target_boxes: torch.Tensor,
        target_labels: torch.Tensor,
    ):
        """Update with new batch."""
        self.predictions.append(
            {"boxes": pred_boxes, "scores": pred_scores, "labels": pred_labels}
        )
        self.targets.append({"boxes": target_boxes, "labels": target_labels})

    def compute(self) -> Dict[str, float]:
        """Compute final metrics."""
        # Placeholder - implement actual mAP calculation
        # In practice, use torchmetrics.MeanAveragePrecision
        return mean_average_precision(
            [p["boxes"] for p in self.predictions],
            [p["scores"] for p in self.predictions],
            [p["labels"] for p in self.predictions],
            [t["boxes"] for t in self.targets],
            [t["labels"] for t in self.targets],
        )


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
        "avg_inference_time": np.mean(times),
        "std_inference_time": np.std(times),
        "min_inference_time": np.min(times),
        "max_inference_time": np.max(times),
        "peak_memory_mb": peak_memory,
        "throughput_fps": 1.0 / np.mean(times),
    }
