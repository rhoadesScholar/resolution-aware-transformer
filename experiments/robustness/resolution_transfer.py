"""Resolution transfer testing - train on one resolution, test on others."""

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import common utilities using relative imports
from ..common.datasets import ISICDataset
from ..common.metrics import SegmentationEvaluator
from ..common.models import load_pretrained_model
from ..common.utils import get_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Resolution transfer testing")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained model"
    )
    parser.add_argument(
        "--config_path", type=str, required=True, help="Path to model config"
    )
    parser.add_argument("--data_dir", type=str, help="Path to dataset")
    parser.add_argument(
        "--output_dir", type=str, default="./results", help="Output directory"
    )
    parser.add_argument(
        "--test_resolutions",
        nargs="+",
        type=int,
        default=[128, 192, 256, 384, 512, 768],
        help="Test resolutions",
    )
    return parser.parse_args()


def evaluate_at_resolution(model, data_dir, resolution, config, device):
    """Evaluate model at specific resolution."""

    # Create dataset at target resolution
    dataset = ISICDataset(
        data_dir=data_dir,
        split="test",
        image_size=resolution,
        multi_scale=config["model"].get("multi_scale", False),
        scales=(
            [resolution]
            if not config["model"].get("multi_scale", False)
            else [resolution, resolution // 2, resolution // 4]
        ),
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=min(16, 64 // (resolution // 128)),  # Adjust batch size for memory
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model.eval()
    evaluator = SegmentationEvaluator()
    inference_times = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating at {resolution}x{resolution}"):
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)

            if config["model"].get("multi_scale", False):
                images = [img.to(device) for img in batch["images"]]
                masks = [mask.to(device) for mask in batch["masks"]]

                start_time.record()
                outputs = model(images)
                end_time.record()

                pred = torch.sigmoid(outputs[0])  # Use highest resolution
                evaluator.update(pred, masks[0])
            else:
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)

                start_time.record()
                outputs = model(images)
                end_time.record()

                pred = torch.sigmoid(outputs)
                evaluator.update(pred, masks)

            torch.cuda.synchronize()
            inference_times.append(start_time.elapsed_time(end_time))

    metrics = evaluator.compute()
    metrics["avg_inference_time"] = np.mean(inference_times)
    metrics["std_inference_time"] = np.std(inference_times)
    metrics["throughput_fps"] = (
        1000.0 / metrics["avg_inference_time"]
    )  # Convert ms to FPS

    return metrics


def plot_resolution_transfer_results(results, output_dir, train_resolution):
    """Plot resolution transfer results."""

    resolutions = list(results.keys())
    dice_scores = [results[res]["dice"] for res in resolutions]
    iou_scores = [results[res]["iou"] for res in resolutions]
    inference_times = [results[res]["avg_inference_time"] for res in resolutions]
    throughput = [results[res]["throughput_fps"] for res in resolutions]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(
        f"Resolution Transfer Analysis\n"
        f"Trained at {train_resolution}x{train_resolution}",
        fontsize=16,
    )

    # Dice scores
    axes[0, 0].plot(resolutions, dice_scores, "o-", linewidth=2, markersize=8)
    axes[0, 0].axvline(
        x=train_resolution,
        color="red",
        linestyle="--",
        alpha=0.7,
        label="Training Resolution",
    )
    axes[0, 0].set_xlabel("Test Resolution")
    axes[0, 0].set_ylabel("Dice Coefficient")
    axes[0, 0].set_title("Dice Score vs Resolution")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # IoU scores
    axes[0, 1].plot(
        resolutions, iou_scores, "o-", linewidth=2, markersize=8, color="orange"
    )
    axes[0, 1].axvline(
        x=train_resolution,
        color="red",
        linestyle="--",
        alpha=0.7,
        label="Training Resolution",
    )
    axes[0, 1].set_xlabel("Test Resolution")
    axes[0, 1].set_ylabel("IoU Score")
    axes[0, 1].set_title("IoU Score vs Resolution")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # Inference time
    axes[1, 0].plot(
        resolutions, inference_times, "o-", linewidth=2, markersize=8, color="green"
    )
    axes[1, 0].set_xlabel("Test Resolution")
    axes[1, 0].set_ylabel("Inference Time (ms)")
    axes[1, 0].set_title("Inference Time vs Resolution")
    axes[1, 0].grid(True, alpha=0.3)

    # Throughput
    axes[1, 1].plot(
        resolutions, throughput, "o-", linewidth=2, markersize=8, color="purple"
    )
    axes[1, 1].set_xlabel("Test Resolution")
    axes[1, 1].set_ylabel("Throughput (FPS)")
    axes[1, 1].set_title("Throughput vs Resolution")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        output_dir / "resolution_transfer_analysis.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Create performance degradation plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Calculate relative performance (normalized to training resolution performance)
    train_idx = resolutions.index(train_resolution)
    train_dice = dice_scores[train_idx]

    relative_performance = [(score / train_dice) * 100 for score in dice_scores]
    scale_factors = [res / train_resolution for res in resolutions]

    ax.plot(scale_factors, relative_performance, "o-", linewidth=2, markersize=8)
    ax.axhline(
        y=100, color="red", linestyle="--", alpha=0.7, label="Training Performance"
    )
    ax.axvline(x=1.0, color="red", linestyle="--", alpha=0.7)
    ax.set_xlabel("Scale Factor (Test Resolution / Training Resolution)")
    ax.set_ylabel("Relative Performance (%)")
    ax.set_title("Performance Degradation vs Scale Factor")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Add acceptable performance band
    ax.fill_between(
        scale_factors,
        90,
        110,
        alpha=0.2,
        color="green",
        label="Acceptable Range (±10%)",
    )
    ax.legend()

    plt.savefig(
        output_dir / "performance_degradation.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def analyze_transfer_patterns(results, train_resolution):
    """Analyze resolution transfer patterns."""

    analysis = {"train_resolution": train_resolution, "transfer_analysis": {}}

    resolutions = sorted(results.keys())
    train_dice = results[train_resolution]["dice"]

    # Analyze upscaling vs downscaling
    upscale_resolutions = [res for res in resolutions if res > train_resolution]
    downscale_resolutions = [res for res in resolutions if res < train_resolution]

    if upscale_resolutions:
        upscale_performance = [results[res]["dice"] for res in upscale_resolutions]
        upscale_degradation = [
            (train_dice - perf) / train_dice * 100 for perf in upscale_performance
        ]
        analysis["transfer_analysis"]["upscaling"] = {
            "resolutions": upscale_resolutions,
            "avg_degradation_percent": np.mean(upscale_degradation),
            "max_degradation_percent": np.max(upscale_degradation),
        }

    if downscale_resolutions:
        downscale_performance = [results[res]["dice"] for res in downscale_resolutions]
        downscale_degradation = [
            (train_dice - perf) / train_dice * 100 for perf in downscale_performance
        ]
        analysis["transfer_analysis"]["downscaling"] = {
            "resolutions": downscale_resolutions,
            "avg_degradation_percent": np.mean(downscale_degradation),
            "max_degradation_percent": np.max(downscale_degradation),
        }

    # Scale invariance metrics
    all_dice = [results[res]["dice"] for res in resolutions]
    analysis["scale_invariance"] = {
        "dice_variance": np.var(all_dice),
        "dice_std": np.std(all_dice),
        "coefficient_of_variation": np.std(all_dice) / np.mean(all_dice),
        "worst_case_degradation": (train_dice - min(all_dice)) / train_dice * 100,
    }

    # Efficiency analysis
    inference_times = [results[res]["avg_inference_time"] for res in resolutions]

    # Calculate computational scaling factor
    time_scaling = []
    for i, res in enumerate(resolutions):
        if res != train_resolution:
            expected_scaling = (res / train_resolution) ** 2  # O(n^2) expected
            actual_scaling = (
                inference_times[i] / results[train_resolution]["avg_inference_time"]
            )
            time_scaling.append(actual_scaling / expected_scaling)

    analysis["efficiency"] = {
        "avg_scaling_factor": np.mean(time_scaling) if time_scaling else 1.0,
        "scaling_efficiency": 1.0 / np.mean(time_scaling) if time_scaling else 1.0,
    }

    return analysis


def main():
    args = parse_args()

    # Load config
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    set_seed(42)
    device = get_device()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading model from: {args.model_path}")

    # Load model
    model_config = config["model"].copy()
    model_name = model_config.pop("name")

    model = load_pretrained_model(
        args.model_path,
        model_name,
        "segmentation",
        1,  # num_classes for segmentation
        **model_config,
    )
    model = model.to(device)

    # Get training resolution from config
    train_resolution = config["data"]["image_size"]
    data_dir = args.data_dir or config["data"]["data_dir"]

    logger.info(f"Training resolution: {train_resolution}x{train_resolution}")
    logger.info(f"Test resolutions: {args.test_resolutions}")

    # Evaluate at each resolution
    results = {}

    for resolution in args.test_resolutions:
        logger.info(f"\nEvaluating at {resolution}x{resolution}...")
        try:
            metrics = evaluate_at_resolution(
                model, data_dir, resolution, config, device
            )
            results[resolution] = metrics

            logger.info(f"  Dice: {metrics['dice']:.4f}")
            logger.info(f"  IoU:  {metrics['iou']:.4f}")
            logger.info(f"  Time: {metrics['avg_inference_time']:.2f}ms")
            logger.info(f"  FPS:  {metrics['throughput_fps']:.2f}")
        except Exception as e:
            logger.info(f"  Error: {e}")
            continue

    if not results:
        logger.info("No successful evaluations!")
        return

    # Analyze results
    analysis = analyze_transfer_patterns(results, train_resolution)

    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info("RESOLUTION TRANSFER ANALYSIS")
    logger.info("=" * 50)

    logger.info(f"Training Resolution: {train_resolution}x{train_resolution}")
    logger.info(f"Baseline Dice: {results[train_resolution]['dice']:.4f}")

    if "upscaling" in analysis["transfer_analysis"]:
        up_analysis = analysis["transfer_analysis"]["upscaling"]
        logger.info("\nUpscaling Performance:")
        logger.info(
            f"  Average degradation: {up_analysis['avg_degradation_percent']:.2f}%"
        )
        logger.info(
            f"  Maximum degradation: {up_analysis['max_degradation_percent']:.2f}%"
        )

    if "downscaling" in analysis["transfer_analysis"]:
        down_analysis = analysis["transfer_analysis"]["downscaling"]
        logger.info("\nDownscaling Performance:")
        logger.info(
            f"  Average degradation: {down_analysis['avg_degradation_percent']:.2f}%"
        )
        logger.info(
            f"  Maximum degradation: {down_analysis['max_degradation_percent']:.2f}%"
        )

    scale_inv = analysis["scale_invariance"]
    logger.info("\nScale Invariance:")
    logger.info(f"  Dice variance: {scale_inv['dice_variance']:.6f}")
    logger.info(
        f"  Coefficient of variation: {scale_inv['coefficient_of_variation']:.4f}"
    )
    logger.info(f"  Worst-case degradation: {scale_inv['worst_case_degradation']:.2f}%")

    efficiency = analysis["efficiency"]
    logger.info("\nComputational Efficiency:")
    logger.info(f"  Scaling efficiency: {efficiency['scaling_efficiency']:.2f}")

    # Save results
    import json

    with open(output_dir / "resolution_transfer_results.json", "w") as f:
        json.dump(
            {"results": results, "analysis": analysis, "config": config}, f, indent=2
        )

    # Create plots
    plot_resolution_transfer_results(results, output_dir, train_resolution)

    logger.info(f"\nResults saved to: {output_dir}")
    logger.info(f"Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
