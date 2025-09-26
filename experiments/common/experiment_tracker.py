#!/usr/bin/env python3
"""
Experiment Results Organization and Tracking System

This module provides a centralized way to organize experiment results with:
- Hierarchical directory structure
- Experiment metadata tracking
- Log aggregation and correlation
- Status-based organization
- Easy cleanup and archival
"""

import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging
import yaml


class ExperimentTracker:
    """
    Centralized experiment tracking and results organization.

    Directory Structure:
    results/
    ├── experiments/
    │   ├── active/           # Currently running experiments
    │   ├── completed/        # Successfully completed experiments
    │   ├── failed/          # Failed experiments with error analysis
    │   └── archived/        # Older experiments moved for cleanup
    ├── logs/
    │   ├── system/          # LSF, SLURM, Ray cluster logs
    │   ├── training/        # Training logs by experiment
    │   └── evaluation/      # Evaluation logs by experiment
    ├── checkpoints/
    │   ├── by_experiment/   # Checkpoints organized by experiment
    │   └── best_models/     # Best performing models across experiments
    ├── tensorboard/         # TensorBoard logs organized by experiment
    └── analysis/            # Analysis reports and comparisons
    """

    def __init__(self, base_results_dir: Union[str, Path] = "results"):
        self.base_dir = Path(base_results_dir).resolve()
        self.setup_directory_structure()

        # Setup logging
        self.logger = self._setup_logging()

        # Track active experiments
        self.registry_path = self.base_dir / "experiment_registry.json"
        self.registry = self._load_registry()

    def setup_directory_structure(self):
        """Create the organized directory structure."""
        dirs_to_create = [
            "experiments/active",
            "experiments/completed",
            "experiments/failed",
            "experiments/archived",
            "logs/system",
            "logs/training",
            "logs/evaluation",
            "checkpoints/by_experiment",
            "checkpoints/best_models",
            "tensorboard",
            "analysis",
        ]

        for dir_path in dirs_to_create:
            (self.base_dir / dir_path).mkdir(parents=True, exist_ok=True)

    def _setup_logging(self) -> logging.Logger:
        """Setup experiment tracker logging."""
        logger = logging.getLogger("experiment_tracker")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.FileHandler(self.base_dir / "experiment_tracker.log")
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _load_registry(self) -> Dict:
        """Load the experiment registry."""
        if self.registry_path.exists():
            with open(self.registry_path, "r") as f:
                return json.load(f)
        return {"experiments": {}, "last_updated": None}

    def _save_registry(self):
        """Save the experiment registry."""
        self.registry["last_updated"] = datetime.now().isoformat()
        with open(self.registry_path, "w") as f:
            json.dump(self.registry, f, indent=2)

    def register_experiment(
        self,
        experiment_name: str,
        config_path: str,
        task_type: str,
        num_gpus: int = 1,
        additional_info: Optional[Dict] = None,
    ) -> str:
        """
        Register a new experiment and create organized directory structure.

        Returns:
            experiment_id: Unique identifier for this experiment run
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_id = f"{experiment_name}_{timestamp}"

        # Create experiment directory in active experiments
        exp_dir = self.base_dir / "experiments" / "active" / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        subdirs = ["logs", "checkpoints", "config", "outputs", "tensorboard"]
        for subdir in subdirs:
            (exp_dir / subdir).mkdir(exist_ok=True)

        # Copy config to experiment directory
        config_source = Path(config_path)
        if config_source.exists():
            shutil.copy2(config_source, exp_dir / "config" / "experiment_config.yaml")

        # Create experiment metadata
        metadata = {
            "experiment_id": experiment_id,
            "experiment_name": experiment_name,
            "task_type": task_type,
            "config_path": str(config_path),
            "num_gpus": num_gpus,
            "start_time": datetime.now().isoformat(),
            "status": "active",
            "directory": str(exp_dir),
            "additional_info": additional_info or {},
        }

        # Save metadata
        with open(exp_dir / "experiment_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Update registry
        self.registry["experiments"][experiment_id] = metadata
        self._save_registry()

        self.logger.info(f"Registered experiment: {experiment_id}")
        return experiment_id

    def update_experiment_status(
        self, experiment_id: str, status: str, additional_info: Optional[Dict] = None
    ):
        """Update experiment status and move to appropriate directory."""
        if experiment_id not in self.registry["experiments"]:
            raise ValueError(f"Experiment {experiment_id} not found in registry")

        exp_metadata = self.registry["experiments"][experiment_id]
        current_dir = Path(exp_metadata["directory"])

        # Update metadata
        exp_metadata["status"] = status
        exp_metadata["end_time"] = datetime.now().isoformat()
        if additional_info:
            exp_metadata["additional_info"].update(additional_info)

        # Move to appropriate directory based on status
        if status in ["completed", "failed"]:
            target_dir = self.base_dir / "experiments" / status / experiment_id
            if current_dir.exists() and current_dir != target_dir:
                target_dir.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(current_dir), str(target_dir))
                exp_metadata["directory"] = str(target_dir)

                # Update metadata in new location
                with open(target_dir / "experiment_metadata.json", "w") as f:
                    json.dump(exp_metadata, f, indent=2)

        # Update registry
        self.registry["experiments"][experiment_id] = exp_metadata
        self._save_registry()

        self.logger.info(f"Updated experiment {experiment_id} status to {status}")

    def organize_existing_results(self, source_dir: Optional[Path] = None):
        """
        Organize existing unstructured results into the new system.
        """
        if source_dir is None:
            source_dir = self.base_dir

        self.logger.info(f"Organizing existing results from {source_dir}")

        # Look for existing experiment directories
        for item in source_dir.iterdir():
            if item.is_dir() and not item.name.startswith(
                (".", "experiments", "logs", "checkpoints", "tensorboard", "analysis")
            ):
                self._migrate_experiment_directory(item)

        # Save the registry after all migrations
        self._save_registry()

    def _migrate_experiment_directory(self, old_dir: Path):
        """Migrate an old experiment directory to the new structure."""
        try:
            # Try to extract experiment info from existing files
            experiment_name = old_dir.name

            # Look for config files
            config_files = list(old_dir.rglob("config.yaml")) + list(
                old_dir.rglob("*.yaml")
            )

            # Look for experiment summary
            summary_files = list(old_dir.rglob("experiment_summary.json"))

            # Determine status from summary or error files
            status = "unknown"
            metadata = {"migrated_from": str(old_dir)}

            if summary_files:
                with open(summary_files[0], "r") as f:
                    summary = json.load(f)
                    status = summary.get("status", "unknown")
                    metadata.update(summary)

            # Check for error files to determine failure
            error_files = list(old_dir.rglob("error.txt")) + list(
                old_dir.rglob("*.err")
            )
            if error_files and status == "unknown":
                status = "failed"

            # Create new experiment entry
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_id = f"{experiment_name}_migrated_{timestamp}"

            # Determine target directory
            target_status = "failed" if status == "failed" else "completed"
            target_dir = self.base_dir / "experiments" / target_status / experiment_id

            # Move the directory
            target_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old_dir), str(target_dir))

            # Create metadata
            exp_metadata = {
                "experiment_id": experiment_id,
                "experiment_name": experiment_name,
                "task_type": "unknown",
                "status": target_status,
                "directory": str(target_dir),
                "migrated": True,
                "migration_time": datetime.now().isoformat(),
                "additional_info": metadata,
            }

            # Save metadata
            with open(target_dir / "experiment_metadata.json", "w") as f:
                json.dump(exp_metadata, f, indent=2)

            # Update registry
            self.registry["experiments"][experiment_id] = exp_metadata

            self.logger.info(f"Migrated experiment: {old_dir.name} -> {experiment_id}")

        except Exception as e:
            self.logger.error(f"Failed to migrate {old_dir}: {e}")

    def get_experiment_summary(self) -> Dict:
        """Get a summary of all experiments."""
        summary = {
            "total_experiments": len(self.registry["experiments"]),
            "by_status": {},
            "recent_experiments": [],
        }

        # Count by status
        for exp in self.registry["experiments"].values():
            status = exp["status"]
            summary["by_status"][status] = summary["by_status"].get(status, 0) + 1

        # Get recent experiments (last 10)
        experiments = sorted(
            self.registry["experiments"].values(),
            key=lambda x: x.get("start_time", ""),
            reverse=True,
        )
        summary["recent_experiments"] = experiments[:10]

        return summary

    def cleanup_old_experiments(self, days_old: int = 30, archive: bool = True):
        """
        Clean up old experiments by archiving or deleting them.

        Args:
            days_old: Experiments older than this many days
            archive: Whether to archive (True) or delete (False)
        """
        cutoff_time = datetime.now().timestamp() - (days_old * 24 * 3600)

        experiments_to_cleanup = []
        for exp_id, exp_data in self.registry["experiments"].items():
            exp_time = datetime.fromisoformat(
                exp_data.get("start_time", "1970-01-01")
            ).timestamp()
            if exp_time < cutoff_time:
                experiments_to_cleanup.append(exp_id)

        for exp_id in experiments_to_cleanup:
            exp_data = self.registry["experiments"][exp_id]
            exp_dir = Path(exp_data["directory"])

            if archive:
                # Move to archived
                archive_dir = self.base_dir / "experiments" / "archived" / exp_id
                archive_dir.parent.mkdir(parents=True, exist_ok=True)
                if exp_dir.exists():
                    shutil.move(str(exp_dir), str(archive_dir))
                    exp_data["directory"] = str(archive_dir)
                    exp_data["status"] = "archived"
                    exp_data["archived_time"] = datetime.now().isoformat()
            else:
                # Delete entirely
                if exp_dir.exists():
                    shutil.rmtree(exp_dir)
                del self.registry["experiments"][exp_id]

        self._save_registry()
        action = "archived" if archive else "deleted"
        self.logger.info(
            f"Cleanup: {action} {len(experiments_to_cleanup)} old experiments"
        )

    def generate_experiment_report(self, output_path: Optional[Path] = None) -> Path:
        """Generate a comprehensive experiment report."""
        if output_path is None:
            output_path = (
                self.base_dir
                / "analysis"
                / f"experiment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            )

        summary = self.get_experiment_summary()

        report_content = f"""# Experiment Report
Generated: {datetime.now().isoformat()}

## Summary
- **Total Experiments**: {summary['total_experiments']}
- **Status Distribution**:
"""

        for status, count in summary["by_status"].items():
            report_content += f"  - {status.title()}: {count}\n"

        report_content += "\n## Recent Experiments\n\n"

        for exp in summary["recent_experiments"]:
            status_emoji = {
                "completed": "✅",
                "failed": "❌",
                "active": "🔄",
                "archived": "📦",
            }.get(exp["status"], "❓")
            report_content += f"- {status_emoji} **{exp['experiment_name']}** ({exp['experiment_id']})\n"
            report_content += f"  - Status: {exp['status']}\n"
            report_content += f"  - Started: {exp.get('start_time', 'Unknown')}\n"
            if exp.get("end_time"):
                report_content += f"  - Ended: {exp['end_time']}\n"
            report_content += f"  - Directory: `{exp['directory']}`\n\n"

        # Write report
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report_content)

        return output_path


def main():
    """CLI interface for experiment tracker."""
    import argparse

    parser = argparse.ArgumentParser(description="Experiment Results Organization Tool")
    parser.add_argument(
        "--results-dir", default="results", help="Base results directory"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Organize command
    organize_parser = subparsers.add_parser(
        "organize", help="Organize existing results"
    )
    organize_parser.add_argument("--source", help="Source directory to organize")

    # Summary command
    subparsers.add_parser("summary", help="Show experiment summary")

    # Report command
    report_parser = subparsers.add_parser("report", help="Generate experiment report")
    report_parser.add_argument("--output", help="Output file path")

    # Cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up old experiments")
    cleanup_parser.add_argument(
        "--days", type=int, default=30, help="Days old threshold"
    )
    cleanup_parser.add_argument(
        "--delete", action="store_true", help="Delete instead of archive"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    tracker = ExperimentTracker(args.results_dir)

    if args.command == "organize":
        source = Path(args.source) if args.source else None
        tracker.organize_existing_results(source)
        print("✅ Results organized successfully")

    elif args.command == "summary":
        summary = tracker.get_experiment_summary()
        print(f"📊 Total Experiments: {summary['total_experiments']}")
        print("\n📈 Status Distribution:")
        for status, count in summary["by_status"].items():
            print(f"  {status.title()}: {count}")

    elif args.command == "report":
        output_path = Path(args.output) if args.output else None
        report_path = tracker.generate_experiment_report(output_path)
        print(f"📄 Report generated: {report_path}")

    elif args.command == "cleanup":
        tracker.cleanup_old_experiments(args.days, not args.delete)
        action = "deleted" if args.delete else "archived"
        print(f"🧹 Old experiments {action}")


if __name__ == "__main__":
    main()
