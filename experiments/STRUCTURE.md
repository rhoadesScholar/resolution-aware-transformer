# Experiments Directory Structure

This directory contains the core training and evaluation scripts for the Resolution Aware Transformer project.

## 🚀 Core Training Scripts

### Primary Training Methods
- **`submit_ray_train.sh`** - **RECOMMENDED** Ray Train with LSF (working solution)
- **`submit_distributed_pytorch.sh`** - Fallback: Direct distributed training without Ray
- **`submit_lsf.sh`** - Advanced: Ray Train with fallback to simple runner

### Python Training Scripts
- **`run_experiment.py`** - Simple experiment runner (single process entry point)
- **`ray_train.py`** - Core Ray Train implementation with DeepSpeed integration
- **`cluster_bootstrap.py`** - Ray cluster bootstrap for LSF environments

## 📊 Evaluation & Analysis
- **`ray_evaluate.py`** - Comprehensive evaluation with robustness testing
- **`organize_results.py`** - Experiment results organization and analysis

## 📁 Directory Structure

### Core Components
- **`common/`** - Shared utilities (datasets, models, metrics, experiment tracking)
- **`configs/`** - YAML configuration files for different experiments
- **`results/`** - Experiment outputs, logs, checkpoints, and tensorboard files

### Specialized Components  
- **`ablations/`** - Ablation study implementations
- **`robustness/`** - Robustness testing and resolution transfer studies

### Data & Checkpoints
- **`data/`** - Dataset storage (populated during experiments)
- **`checkpoints/`** - Model checkpoint storage (populated during training)

## ✅ Recent Fixes Applied

1. **NCCL Timeout Resolution** - Fixed Ray cluster setup and timeout configurations
2. **LSF Integration** - Proper GPU request formatting and distributed training
3. **Path Resolution** - Absolute paths for Ray worker config loading
4. **Simplified Workflow** - Streamlined scripts with working Ray Train setup

## 🎯 Quick Start

```bash
# Recommended approach (Ray Train + LSF)
bsub < experiments/submit_ray_train.sh

# Monitor progress
bpeek <job_id>

# Results will be in:
# - LSF logs: experiments/results/lsf_logs/
# - Ray results: /scratch/$USER/ray_results/
# - Tensorboard: experiments/results/tensorboard/
```

## 📚 Documentation
- **`CLUSTER_DEBUGGING.md`** - Detailed Ray + LSF setup and debugging guide
- **`README.md`** - Comprehensive project documentation