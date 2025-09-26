#!/usr/bin/env python3
"""
Organize Existing Results Script

This script will reorganize the current messy experiment results
into the new structured format with proper tracking.

Usage:
    python organize_results.py
    python organize_results.py --dry-run  # See what would be changed
    python organize_results.py --backup   # Create backup before organizing
"""

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from common.experiment_tracker import ExperimentTracker


def backup_existing_results(results_dir: Path) -> Path:
    """Create a backup of existing results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = results_dir.parent / f"results_backup_{timestamp}"

    print(f"📦 Creating backup: {backup_dir}")
    shutil.copytree(results_dir, backup_dir)
    print(f"✅ Backup created successfully")

    return backup_dir


def analyze_current_structure(results_dir: Path):
    """Analyze the current messy structure and report findings."""
    print("🔍 Analyzing current results structure...")

    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return

    # Find all experiment-related directories
    experiment_dirs = []
    ray_dirs = []
    log_files = []

    for item in results_dir.rglob("*"):
        if item.is_dir():
            if "TorchTrainer_" in item.name:
                ray_dirs.append(item)
            elif item.name.startswith(("rat_", "experiment_", "medical_", "object_")):
                experiment_dirs.append(item)
        elif item.is_file():
            if item.suffix in [".log", ".err", ".out"]:
                log_files.append(item)

    print(f"📊 Analysis Results:")
    print(f"  - Experiment directories found: {len(experiment_dirs)}")
    print(f"  - Ray Train directories found: {len(ray_dirs)}")
    print(f"  - Log files found: {len(log_files)}")

    print(f"\n📁 Experiment Directories:")
    for exp_dir in experiment_dirs[:10]:  # Show first 10
        print(f"  - {exp_dir.relative_to(results_dir)}")
    if len(experiment_dirs) > 10:
        print(f"  ... and {len(experiment_dirs) - 10} more")

    print(f"\n🔄 Ray Train Directories:")
    for ray_dir in ray_dirs[:5]:  # Show first 5
        print(f"  - {ray_dir.relative_to(results_dir)}")
    if len(ray_dirs) > 5:
        print(f"  ... and {len(ray_dirs) - 5} more")

    print(f"\n📜 Log Files:")
    for log_file in log_files[:10]:  # Show first 10
        print(
            f"  - {log_file.relative_to(results_dir)} ({log_file.stat().st_size / 1024:.1f} KB)"
        )
    if len(log_files) > 10:
        print(f"  ... and {len(log_files) - 10} more")


def main():
    parser = argparse.ArgumentParser(
        description="Organize existing messy experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python organize_results.py                    # Organize results
  python organize_results.py --dry-run         # Preview changes
  python organize_results.py --backup          # Create backup first
  python organize_results.py --results-dir ./custom_results  # Custom directory
        """,
    )

    parser.add_argument(
        "--results-dir",
        default="results",
        help="Results directory to organize (default: results)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "--backup", action="store_true", help="Create backup before organizing"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force organization even if target directories exist",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()

    print("🔧 Resolution Aware Transformer - Results Organization Tool")
    print("=" * 60)
    print(f"Target directory: {results_dir}")
    print(f"Dry run mode: {args.dry_run}")
    print(f"Create backup: {args.backup}")
    print("=" * 60)

    # Check if results directory exists
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        print("   Run some experiments first to create results to organize.")
        return

    # Analyze current structure
    analyze_current_structure(results_dir)

    if args.dry_run:
        print("\n🔍 DRY RUN - No changes will be made")
        print("\nProposed organization:")
        print("  results/")
        print("  ├── experiments/")
        print("  │   ├── active/      # Currently running")
        print("  │   ├── completed/   # Successfully finished")
        print("  │   ├── failed/      # Failed experiments")
        print("  │   └── archived/    # Old experiments")
        print("  ├── logs/")
        print("  │   ├── system/      # LSF, SLURM logs")
        print("  │   ├── training/    # Training logs")
        print("  │   └── evaluation/  # Evaluation logs")
        print("  ├── checkpoints/")
        print("  │   ├── by_experiment/")
        print("  │   └── best_models/")
        print("  ├── tensorboard/")
        print("  └── analysis/")
        return

    # Create backup if requested
    if args.backup:
        backup_dir = backup_existing_results(results_dir)
        print(f"📦 Backup created at: {backup_dir}")

    # Ask for confirmation unless force is used
    if not args.force:
        response = input(
            "\n❓ Proceed with reorganization? This will move files around. (y/N): "
        )
        if response.lower() != "y":
            print("❌ Organization cancelled.")
            return

    print("\n🚀 Starting organization...")

    try:
        # Initialize experiment tracker (this creates the new structure)
        tracker = ExperimentTracker(results_dir)

        # Organize existing results
        tracker.organize_existing_results()

        print("✅ Organization completed successfully!")

        # Generate a report of what was done
        report_path = tracker.generate_experiment_report()
        print(f"📄 Organization report saved to: {report_path}")

        # Show summary
        summary = tracker.get_experiment_summary()
        print(f"\n📊 Organization Summary:")
        print(f"  - Total experiments processed: {summary['total_experiments']}")
        print(f"  - Status distribution:")
        for status, count in summary["by_status"].items():
            print(f"    - {status.title()}: {count}")

        print(f"\n🎉 Results are now organized! Use the following commands:")
        print(
            f"  - View summary: python -m common.experiment_tracker summary --results-dir {results_dir}"
        )
        print(
            f"  - Generate report: python -m common.experiment_tracker report --results-dir {results_dir}"
        )
        print(
            f"  - Clean old experiments: python -m common.experiment_tracker cleanup --results-dir {results_dir}"
        )

    except Exception as e:
        print(f"❌ Organization failed: {e}")
        if args.backup:
            print(f"   Your backup is safe at: {backup_dir}")
        raise


if __name__ == "__main__":
    main()
