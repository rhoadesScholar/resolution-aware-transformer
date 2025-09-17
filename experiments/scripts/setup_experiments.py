#!/usr/bin/env python3
"""
Master Setup Script for Resolution Aware Transformer Experiments

This script provides a unified interface to set up all data and configurations
needed for the RAT experiments.

Usage:
    # Quick setup with sample data (for testing)
    python setup_experiments.py --quick

    # Full setup with manual dataset organization
    python setup_experiments.py --isic_downloads /path/to/isic/downloads --coco_downloads /path/to/coco/downloads --data_dir /path/to/data

    # Setup with automatic COCO download
    python setup_experiments.py --isic_downloads /path/to/isic/downloads --download_coco --data_dir /path/to/data
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command with error handling."""
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 50)

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return False


def setup_environment():
    """Set up the Python environment and dependencies."""
    print("Setting up Python environment...")

    # Install the package in development mode
    success = run_command(
        ["pip", "install", "-e", ".[dev]"], "Installing RAT package in development mode"
    )

    if not success:
        print("Warning: Failed to install package. Trying alternative method...")
        run_command(["pip", "install", "-e", "."], "Installing RAT package (basic)")


def setup_sample_data(data_dir):
    """Set up sample datasets for quick testing."""
    scripts_dir = Path(__file__).parent

    print("Setting up sample datasets...")

    # Setup sample ISIC data
    isic_sample_dir = Path(data_dir) / "sample_ISIC2018"
    success = run_command(
        [
            "python",
            str(scripts_dir / "setup_isic.py"),
            "--sample_only",
            "--output_dir",
            str(isic_sample_dir),
            "--num_samples",
            "20",
        ],
        "Creating sample ISIC dataset",
    )

    if not success:
        print("Warning: Failed to create sample ISIC dataset")
        return False

    # Setup sample COCO data
    coco_sample_dir = Path(data_dir) / "sample_COCO2017"
    success = run_command(
        [
            "python",
            str(scripts_dir / "setup_coco.py"),
            "--sample_only",
            "--output_dir",
            str(coco_sample_dir),
            "--num_samples",
            "10",
        ],
        "Creating sample COCO dataset",
    )

    if not success:
        print("Warning: Failed to create sample COCO dataset")
        return False

    return True


def setup_real_datasets(
    data_dir, isic_downloads=None, coco_downloads=None, download_coco=False
):
    """Set up real datasets."""
    scripts_dir = Path(__file__).parent

    success = True

    # Setup ISIC dataset
    if isic_downloads:
        isic_dir = Path(data_dir) / "ISIC2018"
        cmd_success = run_command(
            [
                "python",
                str(scripts_dir / "setup_isic.py"),
                "--downloads_dir",
                isic_downloads,
                "--output_dir",
                str(isic_dir),
            ],
            "Organizing ISIC 2018 dataset",
        )
        if not cmd_success:
            print("Warning: Failed to organize ISIC dataset")
            success = False

    # Setup COCO dataset
    coco_dir = Path(data_dir) / "COCO2017"

    if download_coco:
        cmd_success = run_command(
            [
                "python",
                str(scripts_dir / "setup_coco.py"),
                "--download",
                "--output_dir",
                str(coco_dir),
            ],
            "Downloading and organizing COCO 2017 dataset",
        )
        if not cmd_success:
            print("Warning: Failed to download COCO dataset")
            success = False
    elif coco_downloads:
        cmd_success = run_command(
            [
                "python",
                str(scripts_dir / "setup_coco.py"),
                "--downloads_dir",
                coco_downloads,
                "--output_dir",
                str(coco_dir),
            ],
            "Organizing COCO 2017 dataset",
        )
        if not cmd_success:
            print("Warning: Failed to organize COCO dataset")
            success = False

    return success


def update_configurations(data_dir, quick=False):
    """Update configuration files with correct data paths."""
    scripts_dir = Path(__file__).parent

    print("Updating configuration files...")

    cmd = ["python", str(scripts_dir / "update_configs.py"), "--data_dir", data_dir]

    if quick:
        cmd.append("--create_quick_configs")

    success = run_command(cmd, "Updating configuration files")

    return success


def run_verification_tests(data_dir, quick=False):
    """Run basic verification tests to ensure setup is working."""
    print("Running verification tests...")

    # Test imports
    try:
        print("Testing package imports...")
        import resolution_aware_transformer

        print("✓ Successfully imported resolution_aware_transformer")
    except ImportError as e:
        print(f"✗ Failed to import resolution_aware_transformer: {e}")
        return False

    # Test dataset loading
    try:
        print("Testing dataset loading...")
        from experiments.common.datasets import ISICDataset

        if quick:
            # Test sample dataset
            isic_dir = Path(data_dir) / "sample_ISIC2018"
        else:
            isic_dir = Path(data_dir) / "ISIC2018"

        if isic_dir.exists():
            dataset = ISICDataset(str(isic_dir), split="train")
            print(f"✓ Successfully loaded ISIC dataset with {len(dataset)} samples")
        else:
            print(f"✗ ISIC dataset directory not found: {isic_dir}")
            return False

    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return False

    return True


def print_next_steps(data_dir, quick=False):
    """Print next steps for the user."""
    print("\n" + "=" * 60)
    print("🎉 SETUP COMPLETE! 🎉")
    print("=" * 60)

    if quick:
        print("You've set up RAT with sample datasets for quick testing.")
        print("\nNext steps:")
        print("1. Run a quick test:")
        print("   cd experiments")
        print("   python run_experiments.py --experiments medical_seg --quick")
        print()
        print("2. Run ablation studies:")
        print("   python run_experiments.py --experiments ablations --quick")
        print()
        print("3. When ready for real data, download datasets and re-run:")
        print("   python scripts/setup_experiments.py --real-data")
    else:
        print("You've set up RAT with full datasets.")
        print("\nNext steps:")
        print("1. Run medical segmentation experiments:")
        print("   cd experiments")
        print("   python run_experiments.py --experiments medical_seg")
        print()
        print("2. Run object detection experiments:")
        print("   python run_experiments.py --experiments object_det")
        print()
        print("3. Run complete experimental suite:")
        print("   python run_experiments.py --experiments all")

    print("\nData locations:")
    data_path = Path(data_dir)
    for subdir in data_path.iterdir():
        if subdir.is_dir():
            print(f"  📁 {subdir.name}: {subdir}")

    print("\nConfiguration files updated in:")
    print("  📁 */configs/")

    print("\nFor more options, see:")
    print("  python run_experiments.py --help")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Master setup script for RAT experiments"
    )
    parser.add_argument(
        "--data_dir", type=str, default="./data", help="Base directory for datasets"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Quick setup with sample data for testing"
    )
    parser.add_argument(
        "--isic_downloads",
        type=str,
        help="Directory containing manually downloaded ISIC zip files",
    )
    parser.add_argument(
        "--coco_downloads",
        type=str,
        help="Directory containing manually downloaded COCO zip files",
    )
    parser.add_argument(
        "--download_coco",
        action="store_true",
        help="Automatically download COCO dataset (~25GB)",
    )
    parser.add_argument(
        "--skip_env",
        action="store_true",
        help="Skip environment setup (assume already installed)",
    )
    parser.add_argument(
        "--skip_verification", action="store_true", help="Skip verification tests"
    )

    args = parser.parse_args()

    print("🚀 Starting RAT Experiment Setup")
    print("=" * 50)

    # Create data directory
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Data directory: {data_dir.absolute()}")

    # Step 1: Environment setup
    if not args.skip_env:
        setup_environment()

    # Step 2: Dataset setup
    if args.quick:
        success = setup_sample_data(args.data_dir)
    else:
        if (
            not args.isic_downloads
            and not args.download_coco
            and not args.coco_downloads
        ):
            print("\nError: For real data setup, you must specify:")
            print("  --isic_downloads: Directory with ISIC zip files")
            print("  --coco_downloads: Directory with COCO zip files, OR")
            print("  --download_coco: Automatically download COCO")
            print("\nFor quick testing with sample data, use: --quick")
            sys.exit(1)

        success = setup_real_datasets(
            args.data_dir, args.isic_downloads, args.coco_downloads, args.download_coco
        )

    if not success:
        print("⚠️  Dataset setup completed with warnings. Check messages above.")

    # Step 3: Update configurations
    config_success = update_configurations(args.data_dir, args.quick)

    if not config_success:
        print(
            "⚠️  Configuration update failed. You may need to update configs manually."
        )

    # Step 4: Verification
    if not args.skip_verification:
        verification_success = run_verification_tests(args.data_dir, args.quick)

        if not verification_success:
            print("⚠️  Verification tests failed. Setup may be incomplete.")

    # Step 5: Print next steps
    print_next_steps(args.data_dir, args.quick)


if __name__ == "__main__":
    main()
