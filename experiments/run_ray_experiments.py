#!/usr/bin/env python3
"""
Unified Training and Evaluation Pipeline for Resolution Aware Transformer
Combines ray_train.py and ray_evaluate.py for complete experiment workflows.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import yaml

sys.path.append(str(Path(__file__).parent))
from . import DEFAULT_TEST_RESOLUTIONS

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("experiment.log")],
)
logger = logging.getLogger(__name__)


def find_best_checkpoint(results_dir: Path) -> Optional[str]:
    """Find the best checkpoint from training results."""
    checkpoint_dir = results_dir / "checkpoints"
    if not checkpoint_dir.exists():
        return None

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

    return None


def run_experiment_suite(
    config_path: str,
    experiment_types: List[str],
    num_gpus: int = 4,
    quick_mode: bool = False,
) -> Dict[str, Any]:
    """
    Run a complete experiment suite including training and evaluation.

    Args:
        config_path: Path to configuration file
        experiment_types: List of experiment types to run
        num_gpus: Number of GPUs to use
        quick_mode: Whether to run in quick mode with reduced parameters

    Returns:
        Dictionary with experiment results
    """
    results = {
        "experiment_summary": {
            "config_path": config_path,
            "experiment_types": experiment_types,
            "num_gpus": num_gpus,
            "quick_mode": quick_mode,
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "completed_experiments": [],
            "failed_experiments": [],
        },
        "results": {},
    }

    # Load base configuration
    with open(config_path, "r") as f:
        base_config = yaml.safe_load(f)

    # Create results directory
    results_dir = Path(base_config.get("results", {}).get("output_dir", "./results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    for exp_type in experiment_types:
        logger.info(f"\n{'='*60}")
        logger.info(f"RUNNING EXPERIMENT: {exp_type.upper()}")
        logger.info(f"{'='*60}")

        try:
            exp_results = run_single_experiment(
                config_path, exp_type, num_gpus, quick_mode, results_dir
            )
            results["results"][exp_type] = exp_results
            results["experiment_summary"]["completed_experiments"].append(exp_type)
            logger.info(f"✅ {exp_type} completed successfully")

        except Exception as e:
            logger.error(f"❌ {exp_type} failed: {e}")
            results["experiment_summary"]["failed_experiments"].append(
                {"experiment": exp_type, "error": str(e)}
            )
            continue

    # Generate final report
    results["experiment_summary"]["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    results["experiment_summary"]["total_experiments"] = len(experiment_types)
    results["experiment_summary"]["successful_experiments"] = len(
        results["experiment_summary"]["completed_experiments"]
    )
    results["experiment_summary"]["failed_experiments_count"] = len(
        results["experiment_summary"]["failed_experiments"]
    )

    # Save comprehensive results
    report_path = results_dir / "experiment_suite_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info("EXPERIMENT SUITE COMPLETED")
    logger.info(f"{'='*60}")
    logger.info(
        f"Total experiments: {results['experiment_summary']['total_experiments']}"
    )
    logger.info(
        f"Successful: {results['experiment_summary']['successful_experiments']}"
    )
    logger.info(f"Failed: {results['experiment_summary']['failed_experiments_count']}")
    logger.info(f"Report saved to: {report_path}")

    return results


def run_single_experiment(
    config_path: str, exp_type: str, num_gpus: int, quick_mode: bool, results_dir: Path
) -> Dict[str, Any]:
    """Run a single experiment with training and evaluation."""
    from ray_train import train_rat_with_ray
    from ray_evaluate import evaluate_rat_with_ray

    # Load and modify config for experiment type
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Modify config based on experiment type
    if exp_type == "medical_segmentation":
        config["task_type"] = "segmentation"
        config["data"]["dataset_name"] = "isic2018"
        config["experiment_name"] = f"rat_medical_segmentation"

    elif exp_type == "object_detection":
        config["task_type"] = "detection"
        config["data"]["dataset_name"] = "coco2017"
        config["experiment_name"] = f"rat_object_detection"

    elif exp_type == "ablation_study":
        config["task_type"] = "segmentation"  # Default to segmentation for ablations
        config["data"]["dataset_name"] = "isic2018"
        config["experiment_name"] = f"rat_ablation"

    elif exp_type == "robustness_test":
        config["task_type"] = "segmentation"
        config["data"]["dataset_name"] = "isic2018"
        config["experiment_name"] = f"rat_robustness"

    # Apply quick mode modifications
    if quick_mode:
        config["training"]["epochs"] = min(5, config["training"].get("epochs", 50))
        config["training"]["batch_size"] = min(
            4, config["training"].get("batch_size", 8)
        )
        config["data"]["num_workers"] = 2
        config["experiment_name"] += "_quick"

    # Create experiment-specific results directory
    exp_results_dir = results_dir / exp_type
    exp_results_dir.mkdir(parents=True, exist_ok=True)
    config["results"]["output_dir"] = str(exp_results_dir)

    # Save modified config
    exp_config_path = exp_results_dir / "config.yaml"
    with open(exp_config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Step 1: Training
    logger.info(f"Starting training for {exp_type}")
    start_time = time.time()

    try:
        train_rat_with_ray(
            config_path=str(exp_config_path), num_gpus=num_gpus, num_cpus_per_gpu=4
        )
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time/60:.1f} minutes")

    except Exception as e:
        raise RuntimeError(f"Training failed: {e}")

    # Step 2: Find best checkpoint
    checkpoint_path = find_best_checkpoint(exp_results_dir)
    if not checkpoint_path:
        raise RuntimeError("No checkpoint found after training")

    logger.info(f"Using checkpoint: {checkpoint_path}")

    # Step 3: Evaluation
    logger.info(f"Starting evaluation for {exp_type}")
    eval_start_time = time.time()

    try:
        # Determine evaluation parameters based on experiment type
        robustness_test = exp_type == "robustness_test"

        evaluate_rat_with_ray(
            config_path=str(exp_config_path),
            checkpoint_path=checkpoint_path,
            num_gpus=num_gpus,
            robustness_test=robustness_test,
            test_resolutions=DEFAULT_TEST_RESOLUTIONS,
        )
        evaluation_time = time.time() - eval_start_time
        logger.info(f"Evaluation completed in {evaluation_time/60:.1f} minutes")

    except Exception as e:
        raise RuntimeError(f"Evaluation failed: {e}")

    # Step 4: Collect results
    total_time = time.time() - start_time

    # Load evaluation results
    eval_results_path = exp_results_dir / "evaluation_results.json"
    robustness_results_path = exp_results_dir / "robustness_results.json"

    eval_results = {}
    if eval_results_path.exists():
        with open(eval_results_path, "r") as f:
            eval_results = json.load(f)

    robustness_results = {}
    if robustness_results_path.exists():
        with open(robustness_results_path, "r") as f:
            robustness_results = json.load(f)

    experiment_results = {
        "experiment_type": exp_type,
        "config_path": str(exp_config_path),
        "checkpoint_path": checkpoint_path,
        "training_time_minutes": training_time / 60,
        "evaluation_time_minutes": evaluation_time / 60,
        "total_time_minutes": total_time / 60,
        "evaluation_metrics": eval_results.get("metrics", {}),
        "robustness_metrics": robustness_results,
        "status": "completed",
    }

    return experiment_results


def run_ablation_experiments(
    base_config_path: str, num_gpus: int = 4, quick_mode: bool = False
) -> Dict[str, Any]:
    """
    Run a comprehensive ablation study across multiple model variants.
    """
    logger.info("Starting comprehensive ablation study")

    # Load base configuration
    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f)

    # Define ablation variants
    ablation_variants = {
        "baseline": {},  # Base model
        "no_multiscale": {"model": {"multi_scale": False}},
        "no_rose": {"model": {"positional_encoding": "absolute"}},
        "dense_attention_only": {"model": {"attention_type": "dense"}},
        "sparse_attention_only": {"model": {"attention_type": "sparse"}},
        "small_model": {"model": {"feature_dims": 64, "num_blocks": 2}},
        "large_model": {"model": {"feature_dims": 256, "num_blocks": 6}},
    }

    results = {
        "ablation_summary": {
            "base_config": base_config_path,
            "variants": list(ablation_variants.keys()),
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "results": {},
    }

    # Create ablation results directory
    results_dir = (
        Path(base_config.get("results", {}).get("output_dir", "./results"))
        / "ablation_study"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    for variant_name, variant_config in ablation_variants.items():
        logger.info(f"\nRunning ablation variant: {variant_name}")

        try:
            # Create variant-specific config
            variant_full_config = base_config.copy()

            # Deep merge variant config
            for key, value in variant_config.items():
                if key in variant_full_config and isinstance(value, dict):
                    variant_full_config[key].update(value)
                else:
                    variant_full_config[key] = value

            variant_full_config["experiment_name"] = f"ablation_{variant_name}"

            if quick_mode:
                variant_full_config["training"]["epochs"] = 5
                variant_full_config["training"]["batch_size"] = 4

            # Save variant config
            variant_config_path = results_dir / f"{variant_name}_config.yaml"
            with open(variant_config_path, "w") as f:
                yaml.dump(variant_full_config, f, default_flow_style=False)

            # Run experiment
            variant_results = run_single_experiment(
                str(variant_config_path),
                "ablation_study",
                num_gpus,
                quick_mode,
                results_dir,
            )

            results["results"][variant_name] = variant_results
            logger.info(f"✅ Ablation variant {variant_name} completed")

        except Exception as e:
            logger.error(f"❌ Ablation variant {variant_name} failed: {e}")
            results["results"][variant_name] = {"status": "failed", "error": str(e)}

    # Generate ablation comparison report
    results["ablation_summary"]["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")

    comparison_data = []
    for variant_name, variant_result in results["results"].items():
        if variant_result.get("status") == "completed":
            metrics = variant_result.get("evaluation_metrics", {})
            comparison_data.append({"variant": variant_name, **metrics})

    results["comparison_table"] = comparison_data

    # Save ablation results
    ablation_report_path = results_dir / "ablation_study_report.json"
    with open(ablation_report_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Ablation study completed. Report saved to: {ablation_report_path}")
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="RAT Unified Experiment Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=[
            "medical_segmentation",
            "object_detection",
            "ablation_study",
            "robustness_test",
            "all",
        ],
        default=["all"],
        help="Experiments to run",
    )
    parser.add_argument("--num-gpus", type=int, default=4, help="Number of GPUs")
    parser.add_argument(
        "--quick", action="store_true", help="Quick mode with reduced parameters"
    )
    parser.add_argument(
        "--ablation-only",
        action="store_true",
        help="Run comprehensive ablation study only",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.ablation_only:
        # Run comprehensive ablation study
        results = run_ablation_experiments(
            base_config_path=args.config, num_gpus=args.num_gpus, quick_mode=args.quick
        )
    else:
        # Determine experiments to run
        experiments_to_run = args.experiments
        if "all" in experiments_to_run:
            experiments_to_run = [
                "medical_segmentation",
                "object_detection",
                "ablation_study",
                "robustness_test",
            ]

        # Run experiment suite
        results = run_experiment_suite(
            config_path=args.config,
            experiment_types=experiments_to_run,
            num_gpus=args.num_gpus,
            quick_mode=args.quick,
        )

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 60)

    if args.ablation_only:
        successful_variants = sum(
            1 for r in results["results"].values() if r.get("status") == "completed"
        )
        total_variants = len(results["results"])
        logger.info(
            f"Ablation study: {successful_variants}/{total_variants} variants completed"
        )
    else:
        logger.info(
            f"Experiment suite: {results['experiment_summary']['successful_experiments']}/{results['experiment_summary']['total_experiments']} experiments completed"
        )

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
