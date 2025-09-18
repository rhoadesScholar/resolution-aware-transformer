"""Configuration management for RAT experiments."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
import copy

from .utils import get_available_memory, detect_cluster_environment


@dataclass
class AutoConfig:
    """Auto-configuration based on environment and hardware."""
    
    # Hardware detection
    gpu_memory_gb: float = field(default_factory=lambda: get_available_memory()["gpu_memory_gb"])
    num_gpus: int = field(default_factory=lambda: max(1, torch.cuda.device_count() if torch.cuda.is_available() else 1))
    cluster_type: str = field(default_factory=detect_cluster_environment)
    
    # Model size categories for batch size calculation
    model_size_thresholds: Dict[str, int] = field(default_factory=lambda: {
        "small": 10_000_000,     # < 10M parameters
        "medium": 100_000_000,   # < 100M parameters  
        "large": 1_000_000_000,  # < 1B parameters
        "huge": float("inf"),    # >= 1B parameters
    })
    
    # Batch size recommendations by model size and GPU memory
    batch_size_matrix: Dict[str, Dict[str, int]] = field(default_factory=lambda: {
        "small": {"8gb": 16, "16gb": 32, "24gb": 64, "32gb": 128, "48gb": 128, "80gb": 256},
        "medium": {"8gb": 8, "16gb": 16, "24gb": 32, "32gb": 64, "48gb": 96, "80gb": 128},
        "large": {"8gb": 2, "16gb": 4, "24gb": 8, "32gb": 16, "48gb": 24, "80gb": 32},
        "huge": {"8gb": 1, "16gb": 2, "24gb": 4, "32gb": 8, "48gb": 12, "80gb": 16},
    })
    
    def get_model_size_category(self, num_parameters: int) -> str:
        """Determine model size category based on parameter count."""
        for category, threshold in self.model_size_thresholds.items():
            if num_parameters < threshold:
                return category
        return "huge"
    
    def get_memory_category(self) -> str:
        """Determine GPU memory category."""
        memory = self.gpu_memory_gb
        if memory < 10:
            return "8gb"
        elif memory < 20:
            return "16gb"
        elif memory < 28:
            return "24gb"
        elif memory < 40:
            return "32gb"
        elif memory < 60:
            return "48gb"
        else:
            return "80gb"
    
    def get_recommended_batch_size(self, num_parameters: int) -> int:
        """Get recommended batch size based on model size and available memory."""
        model_size = self.get_model_size_category(num_parameters)
        memory_cat = self.get_memory_category()
        
        batch_size = self.batch_size_matrix[model_size][memory_cat]
        
        # Adjust for multi-GPU setups
        if self.num_gpus > 1:
            # Keep per-GPU batch size reasonable, increase total effective batch size via accumulation
            return max(1, batch_size // 2)
        
        return batch_size
    
    def get_gradient_accumulation_steps(self, target_batch_size: int, per_gpu_batch_size: int) -> int:
        """Calculate gradient accumulation steps to achieve target batch size."""
        effective_batch_size = per_gpu_batch_size * self.num_gpus
        return max(1, target_batch_size // effective_batch_size)


class ConfigManager:
    """Manage experiment configurations with auto-optimization."""
    
    def __init__(self, base_config_dir: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.
        
        Args:
            base_config_dir: Base directory for configurations
        """
        if base_config_dir is None:
            # Default to experiments directory
            self.base_config_dir = Path(__file__).parent.parent
        else:
            self.base_config_dir = Path(base_config_dir)
        
        self.auto_config = AutoConfig()
    
    def load_config(self, config_path: Union[str, Path], auto_optimize: bool = True) -> Dict[str, Any]:
        """
        Load configuration from YAML file with optional auto-optimization.
        
        Args:
            config_path: Path to configuration file
            auto_optimize: Whether to apply auto-optimizations
        
        Returns:
            Configuration dictionary
        """
        config_path = Path(config_path)
        
        # Load base config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Load cluster-specific overrides if they exist
        cluster_config_path = self._get_cluster_config_path(config_path)
        if cluster_config_path.exists():
            with open(cluster_config_path, 'r') as f:
                cluster_config = yaml.safe_load(f)
            config = self._merge_configs(config, cluster_config)
        
        # Apply auto-optimizations
        if auto_optimize:
            config = self._apply_auto_optimizations(config)
        
        # Validate configuration
        self._validate_config(config)
        
        return config
    
    def _get_cluster_config_path(self, config_path: Path) -> Path:
        """Get cluster-specific config path."""
        cluster_type = self.auto_config.cluster_type
        if cluster_type == "local":
            return config_path.parent / f"cluster_{config_path.name}"
        else:
            return config_path.parent / f"{cluster_type}_{config_path.name}"
    
    def _merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge configuration dictionaries."""
        merged = copy.deepcopy(base_config)
        
        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    def _apply_auto_optimizations(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply automatic optimizations based on hardware and environment."""
        optimized = copy.deepcopy(config)
        
        # Estimate model parameters for batch size calculation
        model_config = optimized.get("model", {})
        estimated_params = self._estimate_model_parameters(model_config)
        
        # Auto-optimize batch size if not explicitly set or if requested
        training_config = optimized.setdefault("training", {})
        
        if "batch_size" not in training_config or training_config.get("auto_batch_size", False):
            recommended_batch_size = self.auto_config.get_recommended_batch_size(estimated_params)
            training_config["batch_size"] = recommended_batch_size
            print(f"Auto-selected batch size: {recommended_batch_size}")
        
        # Set gradient accumulation for effective batch size
        target_batch_size = training_config.get("target_batch_size")
        if target_batch_size:
            per_gpu_batch_size = training_config["batch_size"]
            grad_accum_steps = self.auto_config.get_gradient_accumulation_steps(
                target_batch_size, per_gpu_batch_size
            )
            training_config["gradient_accumulation_steps"] = grad_accum_steps
            print(f"Auto-selected gradient accumulation steps: {grad_accum_steps}")
        
        # Auto-enable DeepSpeed Stage 2 for larger models or multi-GPU setups
        if (estimated_params > 50_000_000 or self.auto_config.num_gpus > 1) and not training_config.get("deepspeed", False):
            training_config["deepspeed"] = True
            training_config["zero_stage"] = 2
            print("Auto-enabled DeepSpeed Stage 2")
        
        # Auto-enable mixed precision for modern GPUs
        if torch.cuda.is_available() and not training_config.get("mixed_precision", False):
            # Check for Ampere or newer (compute capability >= 8.0)
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                if props.major >= 8:  # Ampere or newer
                    training_config["mixed_precision"] = True
                    print("Auto-enabled mixed precision for Ampere+ GPU")
                    break
        
        # Optimize data loading
        data_config = optimized.setdefault("data", {})
        if "num_workers" not in data_config:
            # Use 4 workers per GPU, capped at CPU count
            import multiprocessing
            num_workers = min(4 * self.auto_config.num_gpus, multiprocessing.cpu_count())
            data_config["num_workers"] = num_workers
        
        # Set pin_memory for GPU training
        if torch.cuda.is_available() and "pin_memory" not in data_config:
            data_config["pin_memory"] = True
        
        return optimized
    
    def _estimate_model_parameters(self, model_config: Dict[str, Any]) -> int:
        """Estimate number of model parameters from configuration."""
        # Rough estimation based on model type and dimensions
        feature_dims = model_config.get("feature_dims", 128)
        num_blocks = model_config.get("num_blocks", 4)
        num_heads = model_config.get("num_heads", 8)
        
        # Rough estimation for transformer-like models
        # Each transformer block: ~4 * feature_dims^2 parameters
        # Plus input/output projections and embeddings
        params_per_block = 4 * feature_dims * feature_dims
        total_params = num_blocks * params_per_block + feature_dims * 1000  # Add overhead
        
        return int(total_params)
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration for common issues."""
        required_sections = ["model", "training", "data"]
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate training config
        training = config["training"]
        if training.get("batch_size", 1) < 1:
            raise ValueError("batch_size must be >= 1")
        
        if training.get("learning_rate", 1e-4) <= 0:
            raise ValueError("learning_rate must be > 0")
        
        # Validate model config
        model = config["model"]
        if model.get("feature_dims", 128) < 1:
            raise ValueError("feature_dims must be >= 1")
        
        if model.get("num_blocks", 4) < 1:
            raise ValueError("num_blocks must be >= 1")
    
    def create_base_config(
        self,
        model_name: str,
        task_type: str,
        dataset_name: str,
        output_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """
        Create a base configuration template.
        
        Args:
            model_name: Name of the model (e.g., "rat", "baseline")
            task_type: Type of task (e.g., "segmentation", "detection")
            dataset_name: Name of dataset (e.g., "isic2018", "coco")
            output_path: Path to save the configuration
        
        Returns:
            Configuration dictionary
        """
        base_config = {
            "experiment_name": f"{model_name}_{task_type}_{dataset_name}",
            "description": f"{model_name.upper()} model for {task_type} on {dataset_name}",
            "seed": 42,
            
            "model": {
                "name": model_name,
                "spatial_dims": 2,
                "input_features": 3,
                "feature_dims": 128,
                "num_blocks": 4,
                "num_heads": 8,
                "attention_type": "dense",
                "multi_scale": False,
            },
            
            "training": {
                "epochs": 100,
                "auto_batch_size": True,  # Enable auto batch size selection
                "learning_rate": 1e-4,
                "weight_decay": 0.01,
                "scheduler": "cosine",
                "grad_clip": 1.0,
                "save_freq": 10,
                "mixed_precision": True,
            },
            
            "data": {
                "dataset_name": dataset_name,
                "image_size": 256,
                "auto_optimize": True,  # Enable auto data loading optimization
            },
            
            "logging": {
                "backend": "tensorboard",
                "log_dir": "results/tensorboard_logs",
                "use_mlflow": True,
            },
            
            "distributed": {
                "auto_detect": True,  # Auto-detect distributed environment
                "backend": "nccl",
            },
        }
        
        # Apply auto-optimizations
        base_config = self._apply_auto_optimizations(base_config)
        
        # Save to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(base_config, f, default_flow_style=False, indent=2)
        
        print(f"Created base configuration: {output_path}")
        return base_config


# Import torch after defining other classes to avoid circular imports
try:
    import torch
except ImportError:
    torch = None
    print("Warning: PyTorch not available for auto-configuration")


def load_config_with_auto_optimization(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Convenience function to load and auto-optimize a configuration.
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        Optimized configuration dictionary
    """
    manager = ConfigManager()
    return manager.load_config(config_path, auto_optimize=True)