"""Evaluation script for trained models."""

import argparse
from pathlib import Path
import sys

import torch
from tqdm import tqdm
import yaml

# Add common utilities to path
sys.path.append(str(Path(__file__).parent.parent / "common"))
from datasets import ISICDataset
from metrics import SegmentationEvaluator, benchmark_model
from models import load_pretrained_model
from utils import BASELINE_SCORES, calculate_improvement, get_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained segmentation model")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained model"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to model config"
    )
    parser.add_argument("--data_dir", type=str, help="Path to ISIC dataset")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["val", "test"],
        help="Dataset split to evaluate",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./eval_results", help="Output directory"
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="Run speed/memory benchmarks"
    )
    return parser.parse_args()


def evaluate_model(model, dataloader, device, multi_scale=False):
    """Evaluate model on dataset."""
    model.eval()
    evaluator = SegmentationEvaluator()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            if multi_scale:
                images = [img.to(device) for img in batch["images"]]
                masks = [mask.to(device) for mask in batch["masks"]]

                outputs = model(images)
                pred = torch.sigmoid(outputs[0])  # Use highest resolution
                evaluator.update(pred, masks[0])
            else:
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)

                outputs = model(images)
                pred = torch.sigmoid(outputs)
                evaluator.update(pred, masks)

    return evaluator.compute()


def main():
    args = parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    set_seed(config.get("seed", 42))
    device = get_device()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Evaluating model: {args.model_path}")
    print(f"Device: {device}")

    # Load dataset
    data_dir = args.data_dir or config["data"]["data_dir"]
    dataset = ISICDataset(
        data_dir=data_dir,
        split=args.split,
        image_size=config["data"]["image_size"],
        multi_scale=config["model"].get("multi_scale", False),
        scales=config["model"].get("scales", [256, 128, 64]),
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config["data"].get("num_workers", 4),
        pin_memory=True,
    )

    print(f"Dataset: {len(dataset)} samples")

    # Load model
    model_config = config["model"].copy()
    model_name = model_config.pop("name")

    model = load_pretrained_model(
        args.model_path, model_name, "segmentation", 1, **model_config
    )
    model = model.to(device)

    print(
        f"Model parameters: "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    # Evaluate
    print("Running evaluation...")
    metrics = evaluate_model(
        model, dataloader, device, config["model"].get("multi_scale", False)
    )

    # Compare to baselines
    baseline_dice = BASELINE_SCORES["isic2018"]["unet"]["dice"]
    baseline_iou = BASELINE_SCORES["isic2018"]["unet"]["iou"]

    dice_improvement = calculate_improvement(metrics["dice"], baseline_dice)
    iou_improvement = calculate_improvement(metrics["iou"], baseline_iou)

    # Print results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Dice Coefficient: {metrics['dice']:.4f} ± {metrics.get('dice_std', 0):.4f}")
    print(f"IoU Score:        {metrics['iou']:.4f} ± {metrics.get('iou_std', 0):.4f}")
    print(f"Sensitivity:      {metrics['sensitivity']:.4f}")
    print(f"Specificity:      {metrics['specificity']:.4f}")
    print()
    print("COMPARISON TO BASELINE (U-Net):")
    print(f"Dice improvement: {dice_improvement:+.2f}%")
    print(f"IoU improvement:  {iou_improvement:+.2f}%")
    print()

    # Benchmark if requested
    if args.benchmark:
        print("Running speed/memory benchmarks...")

        # Create sample input
        if config["model"].get("multi_scale", False):
            sample_input = [
                torch.randn(1, 3, scale, scale).to(device)
                for scale in config["model"].get("scales", [256, 128, 64])
            ]
        else:
            sample_input = torch.randn(
                1, 3, config["data"]["image_size"], config["data"]["image_size"]
            ).to(device)

        benchmark_results = benchmark_model(model, sample_input)

        print("BENCHMARK RESULTS:")
        print(
            f"Inference time:   "
            f"{benchmark_results['avg_inference_time']*1000:.2f} ± "
            f"{benchmark_results['std_inference_time']*1000:.2f} ms"
        )
        print(f"Throughput:       {benchmark_results['throughput_fps']:.2f} FPS")
        print(f"Peak memory:      {benchmark_results['peak_memory_mb']:.2f} MB")

    # Save detailed results
    detailed_results = {
        "model_path": args.model_path,
        "config": config,
        "metrics": metrics,
        "baseline_comparison": {
            "dice_improvement_percent": dice_improvement,
            "iou_improvement_percent": iou_improvement,
        },
        "dataset_info": {"split": args.split, "num_samples": len(dataset)},
    }

    if args.benchmark:
        detailed_results["benchmark"] = benchmark_results

    # Save to file
    import json

    with open(output_dir / "evaluation_results.json", "w") as f:
        json.dump(detailed_results, f, indent=2)

    print(f"\nDetailed results saved to: {output_dir / 'evaluation_results.json'}")


if __name__ == "__main__":
    main()
