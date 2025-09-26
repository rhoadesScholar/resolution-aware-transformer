#!/usr/bin/env python3
"""
Simple Direct Experiment Runner for Resolution Aware Transformer
Run a single experiment with its dedicated config file.

This is the RECOMMENDED way to run experiments:
- Each config file is self-contained and experiment-specific
- No complex multi-experiment orchestration
- Direct training + evaluation pipeline

Usage:
    python run_experiment.py --config configs/medical_segmentation.yaml --num-gpus 8
    python run_experiment.py --config configs/object_detection.yaml --num-gpus 4 --quick
    python run_experiment.py --config configs/robustness_testing.yaml --evaluation-only --checkpoint results/best_model.pth
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional
import logging
import yaml

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("experiment.log")],
)
logger = logging.getLogger(__name__)


def find_best_checkpoint(results_dir: Path) -> str:
    """Find the best checkpoint from training results."""
    checkpoint_dir = results_dir / "checkpoints"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"No checkpoints directory found at {checkpoint_dir}")

    # Look for best model checkpoint
    best_model_path = checkpoint_dir / "best_model.pth"
    if best_model_path.exists():
        return str(best_model_path)

    # Look for latest checkpoint
    checkpoints = list(checkpoint_dir.glob("checkpoint_epoch_*.pth"))
    if checkpoints:
        # Sort by epoch number
        checkpoints.sort(key=lambda x: int(x.stem.split("_")[-1]))
        return str(checkpoints[-1])

    raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")


def run_experiment(
    config_path: str,
    num_gpus: int = 4,
    quick_mode: bool = False,
    evaluation_only: bool = False,
    checkpoint_path: Optional[str] = None,
    robustness_test: bool = False,
) -> dict:
    """
    Run a complete experiment: training + evaluation.

    Args:
        config_path: Path to experiment config file
        num_gpus: Number of GPUs to use
        quick_mode: Reduce training time for testing
        evaluation_only: Skip training, only run evaluation
        checkpoint_path: Path to checkpoint for evaluation-only mode
        robustness_test: Run robustness evaluation

    Returns:
        Dictionary with experiment results
    """
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    experiment_name = config.get("experiment_name", "rat_experiment")
    task_type = config.get("task_type", "segmentation")

    logger.info(f"Running experiment: {experiment_name}")
    logger.info(f"Task type: {task_type}")
    logger.info(f"Config: {config_path}")
    logger.info(f"GPUs: {num_gpus}")
    logger.info(f"Quick mode: {quick_mode}")
    logger.info(f"Evaluation only: {evaluation_only}")

    # Setup results directory
    results_base_dir = Path(config.get("results", {}).get("output_dir", "./results"))
    results_dir = results_base_dir / experiment_name
    results_dir.mkdir(parents=True, exist_ok=True)

    # Apply quick mode modifications
    if quick_mode:
        config["training"]["epochs"] = min(5, config["training"].get("epochs", 50))
        config["training"]["batch_size"] = min(
            4, config["training"].get("batch_size", 8)
        )
        config["data"]["num_workers"] = 2
        config["experiment_name"] = f"{experiment_name}_quick"

    # Update config with results directory
    config["results"]["output_dir"] = str(results_dir)

    # Save final config
    final_config_path = results_dir / "config.yaml"
    with open(final_config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Initialize results tracking
    experiment_results = {
        "experiment_name": experiment_name,
        "task_type": task_type,
        "config_path": str(final_config_path),
        "num_gpus": num_gpus,
        "quick_mode": quick_mode,
        "evaluation_only": evaluation_only,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    total_start_time = time.time()

    try:
        # Step 1: Training (unless evaluation-only)
        if not evaluation_only:
            logger.info("=" * 60)
            logger.info("STARTING TRAINING")
            logger.info("=" * 60)

            training_start_time = time.time()

            from ray_train import train_rat_with_ray

            train_rat_with_ray(
                config_path=str(final_config_path),
                num_gpus=num_gpus,
                num_cpus_per_gpu=4,
            )

            training_time = time.time() - training_start_time
            experiment_results["training_time_minutes"] = training_time / 60
            logger.info(f"✅ Training completed in {training_time/60:.1f} minutes")

            # Find the checkpoint from training
            try:
                checkpoint_path = find_best_checkpoint(results_dir)
                logger.info(f"Using checkpoint: {checkpoint_path}")
            except FileNotFoundError as ckpt_err:
                logger.error(f"Checkpoint not found after training: {ckpt_err}")
                experiment_results["status"] = "failed"
                experiment_results["error"] = str(ckpt_err)
                experiment_results["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
                summary_path = results_dir / "experiment_summary.json"
                with open(summary_path, "w") as f:
                    json.dump(experiment_results, f, indent=2)
                raise

        else:
            # Use provided checkpoint
            if not checkpoint_path:
                raise ValueError("Checkpoint path required for evaluation-only mode")

            if not Path(checkpoint_path).exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

            experiment_results["training_time_minutes"] = 0
            logger.info(f"Using provided checkpoint: {checkpoint_path}")

        experiment_results["checkpoint_path"] = checkpoint_path

        # Step 2: Evaluation
        logger.info("=" * 60)
        logger.info("STARTING EVALUATION")
        logger.info("=" * 60)

        evaluation_start_time = time.time()

        from ray_evaluate import evaluate_rat_with_ray, DEFAULT_TEST_RESOLUTIONS

        evaluate_rat_with_ray(
            config_path=str(final_config_path),
            checkpoint_path=checkpoint_path,
            num_gpus=num_gpus,
            robustness_test=robustness_test,
            test_resolutions=DEFAULT_TEST_RESOLUTIONS,
        )

        evaluation_time = time.time() - evaluation_start_time
        experiment_results["evaluation_time_minutes"] = evaluation_time / 60
        logger.info(f"✅ Evaluation completed in {evaluation_time/60:.1f} minutes")

        # Step 3: Collect results
        total_time = time.time() - total_start_time
        experiment_results["total_time_minutes"] = total_time / 60

        # Load evaluation results
        eval_results_path = results_dir / "evaluation_results.json"
        robustness_results_path = results_dir / "robustness_results.json"

        if eval_results_path.exists():
            with open(eval_results_path, "r") as f:
                eval_results = json.load(f)
            experiment_results["evaluation_metrics"] = eval_results.get("metrics", {})

        if robustness_results_path.exists():
            with open(robustness_results_path, "r") as f:
                robustness_results = json.load(f)
            experiment_results["robustness_metrics"] = robustness_results

        experiment_results["status"] = "completed"
        experiment_results["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")

        # Save experiment summary
        summary_path = results_dir / "experiment_summary.json"
        with open(summary_path, "w") as f:
            json.dump(experiment_results, f, indent=2)

        logger.info("=" * 60)
        logger.info("EXPERIMENT COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Total time: {total_time/60:.1f} minutes")
        logger.info(f"Results saved to: {results_dir}")
        logger.info(f"Summary saved to: {summary_path}")

        return experiment_results

    except Exception as e:
        experiment_results["status"] = "failed"
        experiment_results["error"] = str(e)
        experiment_results["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")

        # Save failed experiment summary
        summary_path = results_dir / "experiment_summary.json"
        with open(summary_path, "w") as f:
            json.dump(experiment_results, f, indent=2)

        logger.error("=" * 60)
        logger.error("EXPERIMENT FAILED")
        logger.error("=" * 60)
        logger.error(f"Error: {e}")
        raise


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple Direct Experiment Runner for Resolution Aware Transformer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete experiment (training + evaluation):
  python run_experiment.py --config configs/medical_segmentation.yaml --num-gpus 8
  
  # Quick test run:
  python run_experiment.py --config configs/medical_segmentation.yaml --quick --num-gpus 4
  
  # Evaluation only with existing checkpoint:
  python run_experiment.py --config configs/medical_segmentation.yaml --evaluation-only --checkpoint results/best_model.pth --num-gpus 2
  
  # Robustness testing:
  python run_experiment.py --config configs/robustness_testing.yaml --robustness --num-gpus 4
        """,
    )

    parser.add_argument(
        "--config", type=str, required=True, help="Path to experiment config file"
    )
    parser.add_argument("--num-gpus", type=int, default=4, help="Number of GPUs to use")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: reduced epochs and batch size for testing",
    )
    parser.add_argument(
        "--evaluation-only",
        action="store_true",
        help="Skip training, only run evaluation",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file (required for --evaluation-only)",
    )
    parser.add_argument(
        "--robustness",
        action="store_true",
        help="Run robustness evaluation with multiple resolutions",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Validate arguments
    if args.evaluation_only and not args.checkpoint:
        logger.error("--checkpoint is required when using --evaluation-only")
        sys.exit(1)

    if not Path(args.config).exists():
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    try:
        results = run_experiment(
            config_path=args.config,
            num_gpus=args.num_gpus,
            quick_mode=args.quick,
            evaluation_only=args.evaluation_only,
            checkpoint_path=args.checkpoint,
            robustness_test=args.robustness,
        )

        logger.info("SUCCESS: Experiment completed successfully!")
        logger.info(f"Experiment Results: {json.dumps(results, indent=2)}")

    except Exception as e:
        logger.error(f"FAILED: Experiment failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
