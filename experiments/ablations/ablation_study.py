"""Ablation study script for Resolution Aware Transformer."""

import os

# Set OMP_NUM_THREADS to 1 to avoid thread oversubscription in distributed training
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
from pathlib import Path
import sys

import pandas as pd
import torch
from tqdm import tqdm
import yaml

# Add common utilities to path
sys.path.append(str(Path(__file__).parent.parent / "common"))
from datasets import ISICDataset
from metrics import SegmentationEvaluator
from models import create_model, load_pretrained_model
from utils import (
    BASELINE_SCORES,
    ExperimentTracker,
    get_device,
    set_seed,
    setup_logging,
    adjust_config_for_gpu_memory,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run ablation studies")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to ablation config file"
    )
    parser.add_argument("--data_dir", type=str, help="Path to ISIC dataset")
    parser.add_argument(
        "--output_dir", type=str, default="./ablation_results", help="Output directory"
    )
    parser.add_argument(
        "--pretrained_dir", type=str, help="Directory with pretrained models"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Quick evaluation on subset"
    )
    return parser.parse_args()


def evaluate_model(model, dataloader, device, config):
    """Evaluate a model on the validation set."""
    model.eval()
    evaluator = SegmentationEvaluator()

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            if config.get("multi_scale", False):
                images = [img.to(device) for img in batch["images"]]
                masks = [mask.to(device) for mask in batch["masks"]]

                outputs = model(images)

                # Use highest resolution for evaluation
                pred = torch.sigmoid(outputs[0])
                evaluator.update(pred, masks[0])
            else:
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)

                outputs = model(images)
                pred = torch.sigmoid(outputs)
                evaluator.update(pred, masks)

            # Quick evaluation mode
            if config.get("quick", False) and batch_idx >= 10:
                break

    return evaluator.compute()


def run_ablation_study(config, args):
    """Run comprehensive ablation study."""
    device = get_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging (only on rank 0 to avoid duplicate logs)
    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        # Create proper logging directory - use results/experiment_logs instead of checkpoints
        log_dir = output_dir.parent / "results" / "experiment_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        logger = setup_logging(str(log_dir), "ablation_study")
        tracker = ExperimentTracker("ablation_study", str(output_dir))
        tracker.log_config(config)
        tracker.start_timer()
    else:
        logger = None
        tracker = None

    # Load validation dataset
    data_dir = args.data_dir or config["data"]["data_dir"]
    val_dataset = ISICDataset(
        data_dir=data_dir,
        split="val",
        image_size=config["data"]["image_size"],
        multi_scale=False,  # Start with single scale
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config["evaluation"]["batch_size"],
        shuffle=False,
        num_workers=config["data"].get("num_workers", 4),
        pin_memory=True,
    )

    if logger:
        logger.info(f"Validation dataset: {len(val_dataset)} samples")

    # Define ablation configurations
    ablation_configs = []

    # 1. Positional Encoding Ablation
    if config["ablations"].get("positional_encoding", False):
        pe_variants = ["rose", "rope", "absolute", "none"]
        for pe in pe_variants:
            cfg = config["model"]["base"].copy()
            cfg["name"] = "rat"
            cfg["positional_encoding"] = pe
            cfg["learnable_rose"] = pe == "rose"
            ablation_configs.append(
                {"name": f"pe_{pe}", "config": cfg, "category": "positional_encoding"}
            )

    # 2. Attention Type Ablation
    if config["ablations"].get("attention_type", False):
        attention_variants = ["dense", "sparse"]
        for attn in attention_variants:
            cfg = config["model"]["base"].copy()
            cfg["name"] = "rat"
            cfg["attention_type"] = attn
            ablation_configs.append(
                {
                    "name": f"attention_{attn}",
                    "config": cfg,
                    "category": "attention_type",
                }
            )

    # 3. Multi-Resolution Ablation
    if config["ablations"].get("multi_resolution", False):
        # Single resolution
        cfg_single = config["model"]["base"].copy()
        cfg_single["name"] = "rat"
        cfg_single["multi_scale"] = False
        ablation_configs.append(
            {
                "name": "single_resolution",
                "config": cfg_single,
                "category": "multi_resolution",
            }
        )

        # Multi-resolution variants
        scale_variants = [[256, 128], [256, 128, 64], [512, 256, 128]]
        for scales in scale_variants:
            cfg = config["model"]["base"].copy()
            cfg["name"] = "rat"
            cfg["multi_scale"] = True
            cfg["scales"] = scales
            ablation_configs.append(
                {
                    "name": f'multi_res_{"_".join(map(str, scales))}',
                    "config": cfg,
                    "category": "multi_resolution",
                }
            )

    # 4. Architecture Depth Ablation
    if config["ablations"].get("architecture_depth", False):
        depth_variants = [2, 4, 6, 8]
        for depth in depth_variants:
            cfg = config["model"]["base"].copy()
            cfg["name"] = "rat"
            cfg["num_blocks"] = depth
            ablation_configs.append(
                {
                    "name": f"depth_{depth}",
                    "config": cfg,
                    "category": "architecture_depth",
                }
            )

    # 5. Feature Dimension Ablation
    if config["ablations"].get("feature_dims", False):
        dim_variants = [128, 256, 512]
        for dim in dim_variants:
            cfg = config["model"]["base"].copy()
            cfg["name"] = "rat"
            cfg["feature_dims"] = dim

            # Use memory-efficient settings for high dimensions
            if dim >= 512:
                cfg["attention_type"] = (
                    "sparse"  # Use sparse attention for memory efficiency
                )
                cfg["num_blocks"] = 2  # Reduce number of blocks
            elif dim >= 256:
                cfg["num_blocks"] = 3  # Slightly reduce for 256

            ablation_configs.append(
                {"name": f"dim_{dim}", "config": cfg, "category": "feature_dims"}
            )

    if logger:
        logger.info(f"Running {len(ablation_configs)} ablation experiments")

    # Run experiments (only on rank 0 to avoid GPU conflicts)
    rank = int(os.environ.get("RANK", 0))
    if rank != 0:
        if logger:
            logger.info("Non-rank-0 process exiting to avoid GPU conflicts")
        return pd.DataFrame()  # Return empty results for non-rank-0 processes

    results = []

    # Ensure only rank 0 uses GPU for ablations
    device = get_device()
    if logger:
        logger.info(f"Using device: {device} for ablation experiments")

    for exp_config in tqdm(ablation_configs, desc="Ablation experiments"):
        if logger:
            logger.info(f"Running experiment: {exp_config['name']}")

        try:
            # Create model
            model_config = exp_config["config"].copy()
            model_name = model_config.pop("name")

            # Handle multi-scale dataset loading
            if model_config.get("multi_scale", False):
                val_dataset_ms = ISICDataset(
                    data_dir=data_dir,
                    split="val",
                    image_size=config["data"]["image_size"],
                    multi_scale=True,
                    scales=model_config.get("scales", [256, 128, 64]),
                )

                val_loader_ms = torch.utils.data.DataLoader(
                    val_dataset_ms,
                    batch_size=config["evaluation"]["batch_size"]
                    // 2,  # Smaller batch for multi-scale
                    shuffle=False,
                    num_workers=config["data"].get("num_workers", 4),
                    pin_memory=True,
                )
                current_loader = val_loader_ms
                eval_config = model_config
            else:
                # Adjust batch size for memory-intensive experiments
                batch_size = config["evaluation"]["batch_size"]
                feature_dims = model_config.get("feature_dims", 256)

                # Reduce batch size for high-dimensional models
                if feature_dims >= 512:
                    batch_size = max(1, batch_size // 8)  # Much smaller for 512+
                elif feature_dims >= 256:
                    batch_size = max(1, batch_size // 4)  # Smaller for 256+

                val_loader_memory_adjusted = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=config["data"].get("num_workers", 4),
                    pin_memory=True,
                )
                current_loader = val_loader_memory_adjusted
                eval_config = model_config

            model = create_model(
                model_name=model_name,
                task="segmentation",
                num_classes=1,
                **model_config,
            )
            model = model.to(device)

            # Load pretrained weights if available
            if args.pretrained_dir:
                model_path = (
                    Path(args.pretrained_dir) / f"{exp_config['name']}_best.pth"
                )
                if model_path.exists():
                    model = load_pretrained_model(
                        str(model_path), model_name, "segmentation", 1, **model_config
                    )
                    model = model.to(device)
                    if logger:
                        logger.info(f"Loaded pretrained model from {model_path}")
                else:
                    if logger:
                        logger.warning(f"Pretrained model not found: {model_path}")
                    # Skip this experiment if no pretrained model
                    continue

            # Evaluate model
            eval_config["quick"] = args.quick
            try:
                metrics = evaluate_model(model, current_loader, device, eval_config)

                # Validate metrics dictionary
                required_keys = ["dice", "iou", "sensitivity", "specificity"]
                missing_keys = [key for key in required_keys if key not in metrics]
                if missing_keys:
                    if logger:
                        logger.error(
                            f"Missing metrics keys: {missing_keys}, got: {list(metrics.keys())}"
                        )
                    continue

            except Exception as e:
                if logger:
                    logger.error(f"Evaluation failed for {exp_config['name']}: {e}")
                continue

            # Calculate improvement over baseline
            baseline_dice = (
                BASELINE_SCORES.get("isic2018", {}).get("unet", {}).get("dice", 0.847)
            )
            improvement = ((metrics["dice"] - baseline_dice) / baseline_dice) * 100

            result = {
                "experiment": exp_config["name"],
                "category": exp_config["category"],
                "dice": metrics["dice"],
                "iou": metrics["iou"],
                "sensitivity": metrics["sensitivity"],
                "specificity": metrics["specificity"],
                "improvement_over_baseline": improvement,
                "parameters": sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
            }

            # Add model-specific details
            result.update(model_config)

            results.append(result)

            if logger:
                logger.info(
                    f"Results for {exp_config['name']}: "
                    f"Dice={metrics['dice']:.4f}, IoU={metrics['iou']:.4f}"
                )

            # Log to tracker
            if tracker:
                for key, value in result.items():
                    if isinstance(value, (int, float)):
                        tracker.log_metric(f"{exp_config['name']}_{key}", value)

            # Clean up GPU memory
            del model
            torch.cuda.empty_cache()

        except Exception as e:
            if logger:
                logger.error(f"Error in experiment {exp_config['name']}: {str(e)}")
            continue

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "ablation_results.csv", index=False)

    # Generate summary
    summary = generate_ablation_summary(results_df)

    with open(output_dir / "ablation_summary.txt", "w") as f:
        f.write(summary)

    if logger:
        logger.info(f"Ablation study completed. Results saved to {output_dir}")
        logger.info("\n" + summary)

    # Save to tracker
    if tracker:
        tracker.save_results()
        duration = tracker.end_timer()
        if logger:
            logger.info(f"Total time: {duration:.2f} seconds")

    return results_df


def generate_ablation_summary(results_df):
    """Generate a summary of ablation study results."""
    summary = ["Ablation Study Summary", "=" * 50, ""]

    # Overall best results
    best_result = results_df.loc[results_df["dice"].idxmax()]
    summary.append("Best Overall Performance:")
    summary.append(f"  Experiment: {best_result['experiment']}")
    summary.append(f"  Dice: {best_result['dice']:.4f}")
    summary.append(f"  IoU: {best_result['iou']:.4f}")
    summary.append(f"  Improvement: {best_result['improvement_over_baseline']:.2f}%")
    summary.append("")

    # Category-wise analysis
    for category in results_df["category"].unique():
        category_results = results_df[results_df["category"] == category]
        best_in_category = category_results.loc[category_results["dice"].idxmax()]

        summary.append(f"{category.title()} Ablation:")
        summary.append(
            f"  Best: {best_in_category['experiment']} "
            f"(Dice: {best_in_category['dice']:.4f})"
        )

        # Show all results in this category
        for _, row in category_results.iterrows():
            summary.append(f"    {row['experiment']}: {row['dice']:.4f}")
        summary.append("")

    # Parameter efficiency analysis
    summary.append("Parameter Efficiency:")
    results_with_params = results_df.dropna(subset=["parameters"])
    if not results_with_params.empty:
        efficiency = results_with_params["dice"] / (
            results_with_params["parameters"] / 1e6
        )  # Dice per million params
        most_efficient = results_with_params.loc[efficiency.idxmax()]
        summary.append(f"  Most Efficient: {most_efficient['experiment']}")
        summary.append(f"    Dice: {most_efficient['dice']:.4f}")
        summary.append(f"    Parameters: {most_efficient['parameters']:,}")
        summary.append(f"    Efficiency: {efficiency.max():.2f} Dice/M params")

    return "\n".join(summary)


def main():
    args = parse_args()

    # Load and optimize configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Apply dynamic memory optimization
    config = adjust_config_for_gpu_memory(config)

    set_seed(config.get("seed", 42))

    # Run ablation study
    results = run_ablation_study(config, args)

    print("\nAblation study completed!")
    print(f"Results saved to: {args.output_dir}")
    print(f"Best Dice score: {results['dice'].max():.4f}")


if __name__ == "__main__":
    main()
