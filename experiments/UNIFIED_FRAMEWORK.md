# Unified Training Framework Documentation

## Overview

The unified training framework consolidates all RAT experiments into a single, flexible system that automatically handles:

- **Distributed Training**: Auto-detection of SLURM, LSF, or local environments
- **DeepSpeed Stage 2**: Automatic integration for memory optimization
- **Intelligent Batch Sizing**: Automatic calculation based on model size and GPU memory
- **HuggingFace Accelerate**: Simplified distributed training setup
- **MLFlow Integration**: Comprehensive experiment tracking
- **Cluster Support**: Unified launcher for LSF and SLURM

## Key Components

### 1. UnifiedTrainer (`experiments/core/trainer.py`)

The core training class that handles all training scenarios:

```python
from core.trainer import UnifiedTrainer

trainer = UnifiedTrainer(
    config=config,
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    criterion=criterion,
    output_dir="./outputs",
    experiment_name="my_experiment"
)

results = trainer.train()
```

**Features:**
- Auto-detects distributed environment (RANK, WORLD_SIZE, etc.)
- Automatically chooses between Accelerate, DeepSpeed, or standard training
- Handles mixed precision, gradient clipping, and scheduling
- Provides unified interface for all experiment types

### 2. Auto-Configuration (`experiments/core/config.py`)

Intelligent configuration management with hardware-aware optimization:

```python
from core.config import load_config_with_auto_optimization

config = load_config_with_auto_optimization("configs/my_config.yaml")
```

**Auto-Optimizations:**
- **Batch Size**: Calculated based on model parameters and GPU memory
- **Data Loading**: Optimized number of workers and memory pinning
- **DeepSpeed**: Enabled automatically for large models or multi-GPU setups
- **Mixed Precision**: Enabled for Ampere+ GPUs
- **Gradient Accumulation**: Calculated to achieve target batch size

### 3. Unified Training Script (`experiments/unified_train.py`)

Single entry point for all experiments:

```bash
# Local training
python unified_train.py --config configs/unified_template.yaml --data-dir /path/to/data

# With auto-optimization
python unified_train.py --config configs/unified_template.yaml --data-dir /path/to/data --batch-size 8

# Debug mode
python unified_train.py --config configs/unified_template.yaml --data-dir /path/to/data --debug
```

### 4. Cluster Launcher (`experiments/cluster_launcher.py`)

Unified cluster job submission for LSF and SLURM:

```bash
# Auto-detect cluster and submit
python cluster_launcher.py --config configs/unified_template.yaml --experiment-type segmentation --num-gpus 8

# SLURM-specific submission
python cluster_launcher.py --config configs/detection_template.yaml --experiment-type detection --num-gpus 4 --partition gpu

# Dry run (generate scripts without submitting)
python cluster_launcher.py --config configs/unified_template.yaml --experiment-type segmentation --dry-run
```

## Configuration System

### Base Configuration Structure

```yaml
# Experiment metadata
experiment_name: "my_experiment"
description: "Description of the experiment"
task_type: "segmentation"  # or "detection"
seed: 42

# Model configuration
model:
  name: "rat"
  feature_dims: 128
  num_blocks: 4
  multi_scale: false
  # ... other model parameters

# Training with auto-optimization
training:
  epochs: 100
  auto_batch_size: true      # Enable automatic batch size selection
  target_batch_size: 32      # Target effective batch size
  learning_rate: 1e-4
  deepspeed: true            # Auto-enabled DeepSpeed Stage 2
  use_accelerate: true       # Use HuggingFace Accelerate
  mixed_precision: true      # Auto-enabled for Ampere+ GPUs

# Data with auto-optimization
data:
  data_dir: "/path/to/dataset"
  image_size: 256
  auto_optimize: true        # Auto-optimize data loading

# Logging and tracking
logging:
  backend: "tensorboard"
  use_mlflow: true           # Enable MLFlow tracking

# Distributed training (auto-detected)
distributed:
  auto_detect: true
  backend: "nccl"
```

### Environment-Specific Overrides

The system automatically loads cluster-specific configurations:

- `config.yaml` → Base configuration
- `cluster_config.yaml` → Overrides for any cluster
- `slurm_config.yaml` → SLURM-specific overrides
- `lsf_config.yaml` → LSF-specific overrides

## Usage Examples

### 1. Local Development

```bash
# Basic training with auto-optimization
python unified_train.py \
    --config configs/unified_template.yaml \
    --data-dir /path/to/ISIC2018 \
    --output-dir ./outputs

# Override specific parameters
python unified_train.py \
    --config configs/unified_template.yaml \
    --data-dir /path/to/ISIC2018 \
    --batch-size 16 \
    --learning-rate 2e-4 \
    --epochs 50
```

### 2. Multi-GPU Training

```bash
# Auto-detected distributed training
torchrun --nproc_per_node=4 unified_train.py \
    --config configs/unified_template.yaml \
    --data-dir /path/to/ISIC2018

# With Accelerate (preferred)
accelerate launch unified_train.py \
    --config configs/unified_template.yaml \
    --data-dir /path/to/ISIC2018
```

### 3. Cluster Submission

```bash
# Medical segmentation on 8 GPUs
python cluster_launcher.py \
    --config configs/unified_template.yaml \
    --experiment-type segmentation \
    --num-gpus 8 \
    --walltime 24:00 \
    --memory 128

# Object detection on 4 GPUs
python cluster_launcher.py \
    --config configs/detection_template.yaml \
    --experiment-type detection \
    --num-gpus 4 \
    --partition gpu \
    --walltime 12:00:00
```

### 4. Experiment Tracking with MLFlow

```bash
# Enable MLFlow tracking
export MLFLOW_TRACKING_URI="http://localhost:5000"
python unified_train.py \
    --config configs/unified_template.yaml \
    --data-dir /path/to/data

# View results
mlflow ui
```

## Integration with Third-Party Tools

### 1. HuggingFace Accelerate

Automatically handles distributed training, mixed precision, and device placement:

```yaml
training:
  use_accelerate: true
  mixed_precision: true
```

### 2. DeepSpeed Stage 2

Automatic memory optimization for large models:

```yaml
training:
  deepspeed: true
  zero_stage: 2  # Partitions gradients and optimizer states
```

### 3. MLFlow

Comprehensive experiment tracking:

```yaml
logging:
  use_mlflow: true
  mlflow_experiment_name: "rat_experiments"
```

### 4. Ray (Future Integration)

Planned integration for hyperparameter tuning:

```yaml
tuning:
  backend: "ray"
  search_space:
    learning_rate: [1e-5, 1e-3]
    batch_size: [4, 8, 16]
```

## Benefits of the Unified System

### 1. Code Reduction
- **~60% reduction** in duplicate code across experiments
- Single training script for all tasks
- Unified configuration system

### 2. Automatic Optimization
- **Intelligent batch sizing** based on available memory
- **Auto-enable optimizations** (DeepSpeed, mixed precision)
- **Dynamic resource allocation**

### 3. Simplified Deployment
- **Auto-detect environments** (local, SLURM, LSF)
- **Unified cluster submission** scripts
- **Consistent job management**

### 4. Better Experiment Management
- **MLFlow integration** for tracking
- **Standardized configurations**
- **Reproducible experiments**

### 5. Flexibility
- **Override any parameter** via CLI
- **Environment-specific configurations**
- **Easy debugging and development**

## Advanced Usage

### Custom Loss Functions

```python
# Custom loss for specific tasks
class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Custom loss implementation
    
    def forward(self, outputs, targets):
        # Loss calculation
        return loss

# Use with UnifiedTrainer
trainer = UnifiedTrainer(
    config=config,
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    criterion=CustomLoss(),
    output_dir="./outputs"
)
```

### Custom Datasets

```python
# Custom dataset implementation
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, **kwargs):
        # Dataset initialization
    
    def __getitem__(self, idx):
        # Return sample
        return {"image": image, "target": target}

# Use with unified training
train_dataset = CustomDataset(data_dir)
trainer = UnifiedTrainer(
    config=config,
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    criterion=criterion,
    output_dir="./outputs"
)
```

### Environment Variables

```bash
# Override configuration via environment
export RAT_DATA_DIR="/shared/datasets"
export RAT_OUTPUT_DIR="/shared/outputs"
export RAT_BATCH_SIZE="16"

python unified_train.py --config configs/unified_template.yaml
```

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Enable `auto_batch_size: true` in config
   - Reduce `feature_dims` or `num_blocks`
   - Enable DeepSpeed Stage 2: `deepspeed: true`

2. **Slow Data Loading**
   - Enable `auto_optimize: true` in data config
   - Manually set `num_workers` based on CPU count

3. **Distributed Training Issues**
   - Ensure `auto_detect: true` in distributed config
   - Use Accelerate: `use_accelerate: true`
   - Check firewall settings for multi-node training

4. **Cluster Job Failures**
   - Use `--dry-run` to check generated scripts
   - Verify resource requests match cluster limits
   - Check queue/partition availability

### Debugging

```bash
# Enable debug mode
python unified_train.py --config configs/unified_template.yaml --debug

# Verbose logging
export CUDA_LAUNCH_BLOCKING=1
python unified_train.py --config configs/unified_template.yaml

# Check auto-optimizations
python -c "
from core.config import load_config_with_auto_optimization
config = load_config_with_auto_optimization('configs/unified_template.yaml')
print(config)
"
```

## Future Enhancements

1. **Ray Integration**: Hyperparameter tuning support
2. **Weights & Biases**: Additional experiment tracking backend
3. **Auto-Hyperparameter Tuning**: Automatic optimization of learning rates and batch sizes
4. **Multi-Task Training**: Support for joint training across tasks
5. **Model Compression**: Automatic quantization and pruning integration