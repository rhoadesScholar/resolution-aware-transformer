#!/usr/bin/env python3
"""
Experiment Runner for Resolution Aware Transformer
Coordinates and executes all benchmark experiments.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Run RAT benchmark experiments")
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=["medical_seg", "object_det", "ablations", "robustness", "all"],
        default=["all"],
        help="Experiments to run",
    )
    parser.add_argument("--data_dir", type=str, help="Base data directory")
    parser.add_argument("--isic_dir", type=str, help="ISIC dataset directory")
    parser.add_argument("--coco_dir", type=str, help="COCO dataset directory")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./experiment_results",
        help="Output directory",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Quick mode with reduced epochs/data"
    )
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Skip training, only run evaluation",
    )
    parser.add_argument("--gpu", type=str, default="0", help="GPU device to use")
    return parser.parse_args()


def run_command(cmd, cwd=None, capture_output=False):
    """Run shell command with proper error handling."""
    logger.debug(f"Running: {' '.join(cmd)}")
    logger.info(f"Working directory: {cwd or os.getcwd()}")

    try:
        if capture_output:
            result = subprocess.run(
                cmd, cwd=cwd, capture_output=True, text=True, check=True
            )
            return result.stdout, result.stderr
        else:
            subprocess.run(cmd, cwd=cwd, check=True)
            return None, None
    except subprocess.CalledProcessError as e:
        logger.info(f"Command failed with return code {e.returncode}")
        if hasattr(e, "stdout") and e.stdout:
            logger.info(f"STDOUT: {e.stdout}")
        if hasattr(e, "stderr") and e.stderr:
            logger.info(f"STDERR: {e.stderr}")
        raise


def update_config_for_quick_mode(config_path):
    """Update configuration for quick testing."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Reduce epochs for quick testing
    if "training" in config:
        config["training"]["num_epochs"] = min(
            5, config["training"].get("num_epochs", 50)
        )
        config["eval"]["eval_interval"] = 1

    # Save modified config
    quick_config_path = config_path.parent / f"quick_{config_path.name}"
    with open(quick_config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    return quick_config_path


def run_medical_segmentation(args, output_dir):
    """Run medical segmentation experiment."""
    logger.info("\n" + "=" * 50)
    logger.info("RUNNING MEDICAL SEGMENTATION EXPERIMENT")
    logger.info("=" * 50)

    exp_dir = Path(__file__).parent / "medical_segmentation"
    exp_output_dir = output_dir / "medical_segmentation"
    exp_output_dir.mkdir(parents=True, exist_ok=True)

    # Setup environment
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Configure data directory
    isic_dir = args.isic_dir or (args.data_dir and Path(args.data_dir) / "isic2018")
    if not isic_dir:
        logger.info("Warning: No ISIC data directory provided. Using placeholder.")
        isic_dir = "/tmp/isic2018"

    configs = ["single_scale.yaml", "multi_scale.yaml"]

    for config_name in configs:
        logger.info(f"\n--- Running {config_name} ---")

        config_path = exp_dir / "configs" / config_name
        if args.quick:
            config_path = update_config_for_quick_mode(config_path)

        model_output_dir = exp_output_dir / config_name.replace(".yaml", "")
        model_output_dir.mkdir(parents=True, exist_ok=True)

        if not args.skip_training:
            # Training command
            train_cmd = [
                sys.executable,
                "train.py",
                "--config",
                str(config_path),
                "--data_dir",
                str(isic_dir),
                "--output_dir",
                str(model_output_dir),
            ]

            try:
                run_command(train_cmd, cwd=exp_dir)
                logger.info(f"✅ Training completed for {config_name}")
            except subprocess.CalledProcessError:
                logger.info(f"❌ Training failed for {config_name}")
                continue

        # Evaluation (if model exists)
        model_path = model_output_dir / "best_model.pth"
        if model_path.exists():
            eval_cmd = [
                sys.executable,
                "evaluate.py",
                "--model_path",
                str(model_path),
                "--config",
                str(config_path),
                "--data_dir",
                str(isic_dir),
                "--output_dir",
                str(model_output_dir / "evaluation"),
                "--benchmark",
            ]

            try:
                run_command(eval_cmd, cwd=exp_dir)
                logger.info(f"✅ Evaluation completed for {config_name}")
            except subprocess.CalledProcessError:
                logger.info(f"❌ Evaluation failed for {config_name}")


def run_object_detection(args, output_dir):
    """Run object detection experiment."""
    logger.info("\n" + "=" * 50)
    logger.info("RUNNING OBJECT DETECTION EXPERIMENT")
    logger.info("=" * 50)

    exp_dir = Path(__file__).parent / "object_detection"
    exp_output_dir = output_dir / "object_detection"
    exp_output_dir.mkdir(parents=True, exist_ok=True)

    # Configure data directory
    coco_dir = args.coco_dir or (args.data_dir and Path(args.data_dir) / "coco2017")
    if not coco_dir:
        logger.info("Warning: No COCO data directory provided. Using placeholder.")
        coco_dir = "/tmp/coco2017"

    configs = ["single_scale.yaml", "multi_scale.yaml"]

    for config_name in configs:
        logger.info(f"\n--- Running {config_name} ---")

        config_path = exp_dir / "configs" / config_name
        if args.quick:
            config_path = update_config_for_quick_mode(config_path)

        model_output_dir = exp_output_dir / config_name.replace(".yaml", "")
        model_output_dir.mkdir(parents=True, exist_ok=True)

        if not args.skip_training:
            train_cmd = [
                sys.executable,
                "train.py",
                "--config",
                str(config_path),
                "--data_dir",
                str(coco_dir),
                "--output_dir",
                str(model_output_dir),
            ]

            if args.quick:
                train_cmd.append("--debug")

            try:
                run_command(train_cmd, cwd=exp_dir)
                logger.info(f"✅ Training completed for {config_name}")
            except subprocess.CalledProcessError:
                logger.info(f"❌ Training failed for {config_name}")


def run_ablations(args, output_dir):
    """Run ablation studies."""
    logger.info("\n" + "=" * 50)
    logger.info("RUNNING ABLATION STUDIES")
    logger.info("=" * 50)

    exp_dir = Path(__file__).parent / "ablations"
    exp_output_dir = output_dir / "ablations"
    exp_output_dir.mkdir(parents=True, exist_ok=True)

    config_path = exp_dir / "configs" / "ablation.yaml"
    if args.quick:
        config_path = update_config_for_quick_mode(config_path)

    # Setup data directory
    data_dir = args.isic_dir or (args.data_dir and Path(args.data_dir) / "isic2018")
    if not data_dir:
        logger.info("Warning: No data directory provided for ablations.")
        data_dir = "/tmp/isic2018"

    ablation_cmd = [
        sys.executable,
        "ablation_study.py",
        "--config",
        str(config_path),
        "--data_dir",
        str(data_dir),
        "--output_dir",
        str(exp_output_dir),
    ]

    if args.quick:
        ablation_cmd.extend(["--quick_test", "--max_combinations", "4"])

    try:
        run_command(ablation_cmd, cwd=exp_dir)
        logger.info("✅ Ablation studies completed")
    except subprocess.CalledProcessError:
        logger.info("❌ Ablation studies failed")


def run_robustness(args, output_dir):
    """Run robustness testing."""
    logger.info("\n" + "=" * 50)
    logger.info("RUNNING ROBUSTNESS TESTING")
    logger.info("=" * 50)

    exp_dir = Path(__file__).parent / "robustness"
    exp_output_dir = output_dir / "robustness"
    exp_output_dir.mkdir(parents=True, exist_ok=True)

    # Find a trained model to test
    model_paths = []

    # Look for medical segmentation models
    med_seg_dir = output_dir / "medical_segmentation"
    if med_seg_dir.exists():
        for config_dir in med_seg_dir.iterdir():
            if config_dir.is_dir():
                model_path = config_dir / "best_model.pth"
                config_path = (
                    Path(__file__).parent
                    / "medical_segmentation"
                    / "configs"
                    / f"{config_dir.name}.yaml"
                )
                if model_path.exists() and config_path.exists():
                    model_paths.append((model_path, config_path))

    if not model_paths:
        logger.info("❌ No trained models found for robustness testing")
        return

    # Run resolution transfer test on first available model
    model_path, config_path = model_paths[0]
    data_dir = args.isic_dir or (args.data_dir and Path(args.data_dir) / "isic2018")

    resolution_cmd = [
        sys.executable,
        "resolution_transfer.py",
        "--model_path",
        str(model_path),
        "--config_path",
        str(config_path),
        "--data_dir",
        str(data_dir),
        "--output_dir",
        str(exp_output_dir),
    ]

    if args.quick:
        resolution_cmd.extend(["--test_resolutions", "128", "256", "512"])

    try:
        run_command(resolution_cmd, cwd=exp_dir)
        logger.info("✅ Robustness testing completed")
    except subprocess.CalledProcessError:
        logger.info("❌ Robustness testing failed")


def generate_final_report(output_dir):
    """Generate final experiment report."""
    logger.info("\n" + "=" * 50)
    logger.info("GENERATING FINAL REPORT")
    logger.info("=" * 50)

    report = {
        "experiment_summary": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_experiments": 0,
            "successful_experiments": 0,
            "failed_experiments": 0,
        },
        "results": {},
    }

    # Check each experiment directory
    experiments = [
        "medical_segmentation",
        "object_detection",
        "ablations",
        "robustness",
    ]

    for exp_name in experiments:
        exp_dir = output_dir / exp_name
        if exp_dir.exists():
            report["results"][exp_name] = {
                "status": "completed",
                "output_files": [
                    str(f.relative_to(exp_dir))
                    for f in exp_dir.rglob("*")
                    if f.is_file()
                ],
            }
            report["experiment_summary"]["successful_experiments"] += 1
        else:
            report["results"][exp_name] = {"status": "not_run"}
            report["experiment_summary"]["failed_experiments"] += 1

        report["experiment_summary"]["total_experiments"] += 1

    # Save report
    with open(output_dir / "experiment_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    logger.info(f"Total experiments: {report['experiment_summary']['total_experiments']}")
    logger.info(f"Successful: {report['experiment_summary']['successful_experiments']}")
    logger.info(f"Failed: {report['experiment_summary']['failed_experiments']}")
    logger.info(f"\nDetailed report saved to: {output_dir / 'experiment_report.json'}")


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("🚀 Starting Resolution Aware Transformer Benchmark Experiments")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"GPU device: {args.gpu}")
    logger.info(f"Quick mode: {args.quick}")

    # Set GPU environment
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Determine which experiments to run
    experiments_to_run = args.experiments
    if "all" in experiments_to_run:
        experiments_to_run = ["medical_seg", "object_det", "ablations", "robustness"]

    start_time = time.time()

    try:
        # Run experiments
        if "medical_seg" in experiments_to_run:
            run_medical_segmentation(args, output_dir)

        if "object_det" in experiments_to_run:
            run_object_detection(args, output_dir)

        if "ablations" in experiments_to_run:
            run_ablations(args, output_dir)

        if "robustness" in experiments_to_run:
            run_robustness(args, output_dir)

        # Generate final report
        generate_final_report(output_dir)

        end_time = time.time()
        total_time = end_time - start_time

        logger.info("\n🎉 All experiments completed!")
        logger.info(f"Total time: {total_time/3600:.2f} hours")
        logger.info(f"Results saved to: {output_dir}")

    except KeyboardInterrupt:
        logger.info("\n⚠️ Experiments interrupted by user")
    except Exception as e:
        logger.info(f"\n❌ Experiments failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
