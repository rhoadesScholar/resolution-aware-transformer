"""Core training and experimentation framework for RAT experiments."""

from .trainer import UnifiedTrainer
from .config import ConfigManager, AutoConfig
from .utils import (
    auto_detect_distributed,
    calculate_optimal_batch_size,
    setup_accelerate,
    get_available_memory,
)

__all__ = [
    "UnifiedTrainer",
    "ConfigManager", 
    "AutoConfig",
    "auto_detect_distributed",
    "calculate_optimal_batch_size", 
    "setup_accelerate",
    "get_available_memory",
]