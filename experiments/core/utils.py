"""Utility functions for distributed training and resource management."""

import os
import sys
import subprocess
from typing import Dict, Any, Optional, Tuple
import math

import torch
import torch.distributed as dist

# Optional dependencies with graceful fallbacks
try:
    from accelerate import Accelerator
    ACCELERATE_AVAILABLE = True
except ImportError:
    Accelerator = None
    ACCELERATE_AVAILABLE = False

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None
    MLFLOW_AVAILABLE = False

try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    deepspeed = None
    DEEPSPEED_AVAILABLE = False


def auto_detect_distributed() -> Dict[str, Any]:
    """
    Auto-detect distributed training environment from common variables.
    
    Returns:
        Dict containing distributed training configuration:
        - is_distributed: bool
        - rank: int (global rank)
        - local_rank: int (local rank within node)
        - world_size: int (total number of processes)
        - backend: str (communication backend)
        - master_addr: str
        - master_port: str
    """
    env_vars = os.environ
    
    # Check for various distributed training setups
    is_distributed = False
    rank = 0
    local_rank = 0
    world_size = 1
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    master_addr = "localhost"
    master_port = "12355"
    
    # PyTorch distributed launch
    if "RANK" in env_vars and "WORLD_SIZE" in env_vars:
        is_distributed = True
        rank = int(env_vars["RANK"])
        world_size = int(env_vars["WORLD_SIZE"])
        local_rank = int(env_vars.get("LOCAL_RANK", 0))
        master_addr = env_vars.get("MASTER_ADDR", "localhost")
        master_port = env_vars.get("MASTER_PORT", "12355")
    
    # SLURM environment
    elif "SLURM_PROCID" in env_vars:
        is_distributed = True
        rank = int(env_vars["SLURM_PROCID"])
        world_size = int(env_vars["SLURM_NTASKS"])
        local_rank = int(env_vars.get("SLURM_LOCALID", 0))
        
        # Get SLURM node list and set master address
        if "SLURM_NODELIST" in env_vars:
            # Extract first node as master
            import re
            nodelist = env_vars["SLURM_NODELIST"]
            match = re.search(r"([a-zA-Z0-9\-]+)", nodelist)
            if match:
                master_addr = match.group(1)
    
    # LSF environment
    elif "LSB_JOBID" in env_vars and "LSB_HOSTS" in env_vars:
        is_distributed = True
        hosts = env_vars["LSB_HOSTS"].split()
        world_size = len(hosts)
        # For LSF, we need to determine rank based on hostname
        import socket
        hostname = socket.gethostname()
        try:
            rank = hosts.index(hostname)
        except ValueError:
            rank = 0
        local_rank = rank % torch.cuda.device_count() if torch.cuda.is_available() else 0
        master_addr = hosts[0] if hosts else "localhost"
    
    # Single GPU check
    elif torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Multiple GPUs detected ({torch.cuda.device_count()}) but no distributed env. "
              "Consider using torchrun for multi-GPU training.")
    
    return {
        "is_distributed": is_distributed,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "backend": backend,
        "master_addr": master_addr,
        "master_port": master_port,
    }


def get_available_memory() -> Dict[str, float]:
    """
    Get available GPU and system memory.
    
    Returns:
        Dict with 'gpu_memory_gb' and 'system_memory_gb' keys
    """
    memory_info = {"gpu_memory_gb": 0.0, "system_memory_gb": 0.0}
    
    # Get GPU memory
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        total_gpu_memory = 0
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            total_gpu_memory += props.total_memory
        memory_info["gpu_memory_gb"] = total_gpu_memory / (1024**3)
    
    # Get system memory
    try:
        import psutil
        memory_info["system_memory_gb"] = psutil.virtual_memory().total / (1024**3)
    except ImportError:
        # Fallback to /proc/meminfo on Linux
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        mem_kb = int(line.split()[1])
                        memory_info["system_memory_gb"] = mem_kb / (1024**2)
                        break
        except (FileNotFoundError, ValueError):
            memory_info["system_memory_gb"] = 8.0  # Conservative fallback
    
    return memory_info


def calculate_optimal_batch_size(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    available_memory_gb: float,
    safety_factor: float = 0.8,
    min_batch_size: int = 1,
    max_batch_size: int = 128,
) -> int:
    """
    Calculate optimal batch size based on model size and available memory.
    
    Args:
        model: PyTorch model
        input_shape: Shape of a single input sample (without batch dimension)
        available_memory_gb: Available GPU memory in GB
        safety_factor: Safety factor to prevent OOM (0.8 = use 80% of memory)
        min_batch_size: Minimum allowed batch size
        max_batch_size: Maximum allowed batch size
    
    Returns:
        Optimal batch size
    """
    if not torch.cuda.is_available():
        return min_batch_size
    
    # Estimate model memory usage
    model_params = sum(p.numel() for p in model.parameters())
    
    # Rough estimates (in bytes):
    # - Parameters: 4 bytes per parameter (float32)
    # - Gradients: same as parameters
    # - Optimizer states: 2x parameters for Adam (momentum + variance)
    # - Activations: highly variable, use heuristic
    
    bytes_per_param = 4  # float32
    model_memory_bytes = model_params * bytes_per_param
    gradient_memory_bytes = model_memory_bytes  # Same as parameters
    optimizer_memory_bytes = model_memory_bytes * 2  # Adam states
    
    # Estimate activation memory per sample (heuristic)
    # This is a rough estimate based on typical transformer models
    input_size = 1
    for dim in input_shape:
        input_size *= dim
    
    # Rough estimate: activations are ~10x input size for transformers
    activation_per_sample_bytes = input_size * bytes_per_param * 10
    
    # Total fixed memory
    fixed_memory_bytes = model_memory_bytes + gradient_memory_bytes + optimizer_memory_bytes
    
    # Available memory for activations
    available_memory_bytes = available_memory_gb * (1024**3) * safety_factor
    available_for_activations = available_memory_bytes - fixed_memory_bytes
    
    if available_for_activations <= 0:
        print(f"Warning: Model too large for available memory. Using minimum batch size.")
        return min_batch_size
    
    # Calculate max batch size based on activation memory
    max_batch_from_memory = int(available_for_activations / activation_per_sample_bytes)
    
    # Apply constraints
    optimal_batch_size = max(min_batch_size, min(max_batch_from_memory, max_batch_size))
    
    # Round down to nearest power of 2 for better performance
    optimal_batch_size = 2 ** int(math.log2(optimal_batch_size))
    
    print(f"Calculated optimal batch size: {optimal_batch_size}")
    print(f"  Model parameters: {model_params:,}")
    print(f"  Available GPU memory: {available_memory_gb:.1f} GB")
    print(f"  Fixed memory usage: {fixed_memory_bytes / (1024**3):.1f} GB")
    
    return max(optimal_batch_size, min_batch_size)


def setup_accelerate(config: Dict[str, Any]) -> Optional[Accelerator]:
    """
    Setup HuggingFace Accelerate if available.
    
    Args:
        config: Training configuration
    
    Returns:
        Accelerator instance or None if not available
    """
    if not ACCELERATE_AVAILABLE:
        print("Warning: Accelerate not available. Falling back to manual distributed setup.")
        return None
    
    try:
        accelerator = Accelerator(
            mixed_precision="fp16" if config.get("mixed_precision", False) else "no",
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
            cpu=not torch.cuda.is_available(),
        )
        
        print(f"Accelerate setup successful:")
        print(f"  Device: {accelerator.device}")
        print(f"  Distributed: {accelerator.distributed_type}")
        print(f"  Mixed precision: {accelerator.mixed_precision}")
        print(f"  Num processes: {accelerator.num_processes}")
        
        return accelerator
    
    except Exception as e:
        print(f"Warning: Failed to setup Accelerate: {e}")
        return None


def setup_deepspeed_stage2(
    config: Dict[str, Any],
    model: torch.nn.Module,
    world_size: int = 1,
) -> Dict[str, Any]:
    """
    Create DeepSpeed Stage 2 configuration.
    
    Args:
        config: Training configuration
        model: PyTorch model
        world_size: Number of distributed processes
    
    Returns:
        DeepSpeed configuration dictionary
    """
    batch_size = config.get("batch_size", 4)
    learning_rate = config.get("learning_rate", 1e-4)
    
    # Calculate effective batch size
    gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)
    train_batch_size = batch_size * world_size * gradient_accumulation_steps
    
    ds_config = {
        "train_batch_size": train_batch_size,
        "train_micro_batch_size_per_gpu": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": learning_rate,
                "weight_decay": config.get("weight_decay", 0.01),
                "betas": [0.9, 0.999],
                "eps": 1e-8,
            },
        },
        
        "scheduler": {
            "type": "WarmupCosineLR",
            "params": {
                "total_num_steps": config.get("total_steps", 10000),
                "warmup_num_steps": config.get("warmup_steps", 100),
            },
        } if config.get("scheduler") == "cosine" else None,
        
        "fp16": {
            "enabled": config.get("mixed_precision", True),
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1,
        },
        
        "gradient_clipping": config.get("grad_clip", 1.0),
        
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True,
            "cpu_offload": False,  # Stage 2 typically doesn't need CPU offload
        },
        
        "activation_checkpointing": {
            "partition_activations": True,
            "cpu_checkpointing": False,
            "contiguous_memory_optimization": False,
            "number_checkpoints": 4,
            "synchronize_checkpoint_boundary": False,
            "profile": False,
        },
        
        "wall_clock_breakdown": False,
        "dump_state": False,
    }
    
    # Remove None values
    ds_config = {k: v for k, v in ds_config.items() if v is not None}
    
    return ds_config


def setup_mlflow_tracking(config: Dict[str, Any], experiment_name: str) -> None:
    """
    Setup MLFlow experiment tracking if available.
    
    Args:
        config: Training configuration
        experiment_name: Name of the experiment
    """
    if not MLFLOW_AVAILABLE:
        print("Warning: MLFlow not available. Skipping experiment tracking.")
        return
    
    try:
        # Set experiment
        mlflow.set_experiment(experiment_name)
        
        # Start run
        with mlflow.start_run():
            # Log hyperparameters
            mlflow.log_params({
                "model_name": config.get("model", {}).get("name", "unknown"),
                "batch_size": config.get("batch_size", 4),
                "learning_rate": config.get("learning_rate", 1e-4),
                "epochs": config.get("epochs", 100),
                "scheduler": config.get("scheduler", "none"),
                "mixed_precision": config.get("mixed_precision", False),
            })
            
            # Log model config
            model_config = config.get("model", {})
            for key, value in model_config.items():
                if isinstance(value, (int, float, str, bool)):
                    mlflow.log_param(f"model_{key}", value)
        
        print(f"MLFlow tracking setup for experiment: {experiment_name}")
    
    except Exception as e:
        print(f"Warning: Failed to setup MLFlow tracking: {e}")


def detect_cluster_environment() -> str:
    """
    Detect the cluster environment (LSF, SLURM, or local).
    
    Returns:
        String indicating the cluster type: "lsf", "slurm", or "local"
    """
    if "LSB_JOBID" in os.environ:
        return "lsf"
    elif "SLURM_JOB_ID" in os.environ:
        return "slurm"
    else:
        return "local"


def get_cluster_job_info() -> Dict[str, Any]:
    """
    Get job information from cluster environment.
    
    Returns:
        Dictionary with job information
    """
    cluster_type = detect_cluster_environment()
    job_info = {"cluster_type": cluster_type}
    
    if cluster_type == "lsf":
        job_info.update({
            "job_id": os.environ.get("LSB_JOBID"),
            "job_name": os.environ.get("LSB_JOBNAME"),
            "queue": os.environ.get("LSB_QUEUE"),
            "hosts": os.environ.get("LSB_HOSTS", "").split(),
        })
    elif cluster_type == "slurm":
        job_info.update({
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "job_name": os.environ.get("SLURM_JOB_NAME"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "num_nodes": int(os.environ.get("SLURM_JOB_NUM_NODES", 1)),
            "num_tasks": int(os.environ.get("SLURM_NTASKS", 1)),
        })
    
    return job_info