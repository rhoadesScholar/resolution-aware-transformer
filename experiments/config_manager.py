#!/usr/bin/env python3
"""
RAT Experiments Configuration Manager

This module handles reading and validating configuration settings for RAT experiments.
It supports both local development and cluster deployment configurations.
"""

import os
import configparser
from pathlib import Path
from typing import Dict, Any, Optional
import sys


class RATConfig:
    """Configuration manager for RAT experiments."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to config file. If None, looks for .config in experiments folder.
        """
        if config_path is None:
            # Find experiments/.config relative to this script
            script_dir = Path(__file__).parent
            self.config_path = script_dir / ".config"
        else:
            self.config_path = Path(config_path)
        self.config = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
        self._load_config()
        self._validate_config()

    def _load_config(self):
        """Load configuration from file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        self.config.read(self.config_path)

    def _validate_config(self):
        """Validate required configuration sections and keys."""
        required_sections = [
            "cluster",
            "paths",
            "datasets",
            "training",
            "logging",
            "experiments",
        ]

        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")

    def _expand_path(self, path: str) -> Path:
        """Expand relative paths based on repo root."""
        path_obj = Path(path)
        if path_obj.is_absolute():
            return path_obj
        else:
            repo_root = Path(self.get("paths", "repo_root"))
            return repo_root / path_obj

    def get(self, section: str, key: str, fallback: Any = None) -> Any:
        """Get configuration value with type conversion."""
        try:
            value = self.config.get(section, key)

            # Handle boolean values
            if value.lower() in ["true", "false"]:
                return value.lower() == "true"

            # Handle integer values
            if value.isdigit():
                return int(value)

            # Handle empty strings as None
            if value.strip() == "":
                return None

            return value
        except (configparser.NoSectionError, configparser.NoOptionError):
            return fallback

    def get_path(self, section: str, key: str) -> Path:
        """Get path configuration value and expand it."""
        path_str = self.get(section, key)
        if path_str is None:
            raise ValueError(f"Path not configured: {section}.{key}")
        return self._expand_path(path_str)

    def get_cluster_config(self) -> Dict[str, Any]:
        """Get cluster-specific configuration."""
        return {
            "queue": self.get("cluster", "queue"),
            "walltime_hours": self.get(
                "cluster", "walltime_hours", None
            ),  # Optional - no time limit if None
            "walltime_hours_quick": self.get(
                "cluster", "walltime_hours_quick", None
            ),  # Optional
            "num_gpus_full": self.get("cluster", "num_gpus_full"),
            "num_gpus_quick": self.get("cluster", "num_gpus_quick"),
            "gpu_mode": self.get("cluster", "gpu_mode"),
            "cpus_per_gpu": self.get("cluster", "cpus_per_gpu"),
            "memory_mb_per_gpu": self.get("cluster", "memory_mb_per_gpu"),
            "span_hosts": self.get("cluster", "span_hosts"),
            "exclusive_node": self.get("cluster", "exclusive_node"),
        }

    def get_paths_config(self) -> Dict[str, Path]:
        """Get all path configurations."""
        paths = {}
        path_keys = ["repo_root", "data_dir", "results_dir", "checkpoints_dir"]

        for key in path_keys:
            paths[key] = self.get_path("paths", key)

        # Add logging subdirectories
        results_dir = paths["results_dir"]
        paths["lsf_logs_dir"] = results_dir / self.get("paths", "lsf_logs_subdir")
        paths["tensorboard_logs_dir"] = results_dir / self.get(
            "paths", "tensorboard_logs_subdir"
        )
        paths["experiment_logs_dir"] = results_dir / self.get(
            "paths", "experiment_logs_subdir"
        )

        return paths

    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration."""
        return {
            "mixed_precision": self.get("training", "mixed_precision"),
            "distributed_backend": self.get("training", "distributed_backend"),
            "find_unused_parameters": self.get("training", "find_unused_parameters"),
            "master_port_base": self.get("training", "master_port_base"),
            "port_increment": self.get("training", "port_increment"),
        }

    def get_dataset_config(self) -> Dict[str, Any]:
        """Get dataset configuration."""
        return {
            "isic_source_dir": self.get("datasets", "isic_source_dir"),
            "coco_source_dir": self.get("datasets", "coco_source_dir"),
            "use_sample_data": self.get("datasets", "use_sample_data"),
            "isic_sample_size": self.get("datasets", "isic_sample_size"),
            "coco_sample_size": self.get("datasets", "coco_sample_size"),
        }

    def get_experiments_config(self) -> Dict[str, Any]:
        """Get experiments configuration."""
        return {
            "medical_segmentation": self.get("experiments", "medical_segmentation"),
            "object_detection": self.get("experiments", "object_detection"),
            "ablation_studies": self.get("experiments", "ablation_studies"),
            "robustness_tests": self.get("experiments", "robustness_tests"),
            "parallel_jobs": self.get("experiments", "parallel_jobs", False),
            "job_dependencies": self.get("experiments", "job_dependencies", True),
            "medical_seg_epochs": self.get("experiments", "medical_seg_epochs"),
            "object_det_epochs": self.get("experiments", "object_det_epochs"),
            "ablation_epochs": self.get("experiments", "ablation_epochs"),
        }

    def get_lsf_job_config(self, quick_test: bool = False) -> Dict[str, Any]:
        """Get LSF job configuration."""
        cluster_config = self.get_cluster_config()
        paths_config = self.get_paths_config()

        if quick_test:
            num_gpus = cluster_config["num_gpus_quick"]
            walltime_hours = cluster_config["walltime_hours_quick"]
            job_name = f"{self.get('environment', 'job_name_prefix')}_quick_test"
        else:
            num_gpus = cluster_config["num_gpus_full"]
            walltime_hours = cluster_config["walltime_hours"]
            job_name = f"{self.get('environment', 'job_name_prefix')}_experiments"

        total_cpus = num_gpus * cluster_config["cpus_per_gpu"]
        total_memory = num_gpus * cluster_config["memory_mb_per_gpu"]

        config = {
            "job_name": job_name,
            "queue": cluster_config["queue"],
            "num_gpus": num_gpus,
            "total_cpus": total_cpus,
            "total_memory": total_memory,
            "walltime_hours": walltime_hours,  # Can be None for no time limit
            "gpu_mode": cluster_config["gpu_mode"],
            "span_hosts": cluster_config["span_hosts"],
            "exclusive_node": cluster_config["exclusive_node"],
            "output_file": paths_config["lsf_logs_dir"] / f"{job_name}_%J.out",
            "error_file": paths_config["lsf_logs_dir"] / f"{job_name}_%J.err",
        }

        return config

    def create_directories(self):
        """Create necessary directories based on configuration."""
        paths = self.get_paths_config()

        directories_to_create = [
            paths["data_dir"],
            paths["results_dir"],
            paths["checkpoints_dir"],
            paths["lsf_logs_dir"],
            paths["tensorboard_logs_dir"],
            paths["experiment_logs_dir"],
        ]

        created = []
        for directory in directories_to_create:
            if not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)
                created.append(directory)

        return created

    def get_local_data_dir(self, job_id: Optional[str] = None) -> Path:
        """Get local temporary data directory."""
        if self.get("paths", "use_local_storage"):
            base_dir = Path(self.get("paths", "local_temp_dir"))
            if job_id:
                return base_dir / f"rat_data_{job_id}"
            else:
                return base_dir / "rat_data"
        else:
            return self.get_path("paths", "data_dir")

    def dump_config(self) -> str:
        """Dump current configuration as a formatted string."""
        output = []
        output.append("RAT Experiments Configuration:")
        output.append("=" * 40)

        for section_name in self.config.sections():
            output.append(f"\n[{section_name}]")
            for key, value in self.config[section_name].items():
                output.append(f"  {key} = {value}")

        return "\n".join(output)


def load_config(config_path: Optional[str] = None) -> RATConfig:
    """Load RAT configuration."""
    return RATConfig(config_path)


if __name__ == "__main__":
    """CLI interface for configuration management."""
    import argparse

    parser = argparse.ArgumentParser(description="RAT Configuration Manager")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument(
        "--dump", action="store_true", help="Dump current configuration"
    )
    parser.add_argument(
        "--create-dirs", action="store_true", help="Create configured directories"
    )
    parser.add_argument(
        "--validate", action="store_true", help="Validate configuration"
    )
    parser.add_argument(
        "--lsf-config", action="store_true", help="Show LSF job configuration"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Show quick test configuration"
    )

    args = parser.parse_args()

    try:
        config = load_config(args.config)

        if args.dump:
            print(config.dump_config())

        if args.validate:
            print("✓ Configuration is valid")

        if args.create_dirs:
            created = config.create_directories()
            if created:
                print("Created directories:")
                for directory in created:
                    print(f"  {directory}")
            else:
                print("All directories already exist")

        if args.lsf_config:
            lsf_config = config.get_lsf_job_config(quick_test=args.quick)
            print("\nLSF Job Configuration:")
            for key, value in lsf_config.items():
                print(f"  {key}: {value}")

        if not any([args.dump, args.validate, args.create_dirs, args.lsf_config]):
            print("RAT configuration loaded successfully")
            print(f"Config file: {config.config_path}")
            print("Use --help for available options")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
