# Resolution Aware Transformer - AI Coding Instructions

## Project Overview
The Resolution Aware Transformer (RAT) is a PyTorch implementation for multi-scale image analysis, particularly suited for microscopy and medical imaging. The architecture combines **Spatial Grouping Attention (SGA)**, **Rotary Spatial Embeddings (RoSE)**, and **Multi-resolution Processing** for handling image pyramids with different resolutions.

## Core Architecture Components

### Multi-Scale Processing Pipeline
- **Input**: Single images or image pyramids at multiple resolutions (e.g., 256x256, 128x128, 64x64)
- **Output**: Embeddings with spatial awareness and attention maps per scale
- **Key Classes**: `ResolutionAwareTransformer` in `src/resolution_aware_transformer/resolution_aware_transformer.py`

### Dependencies & Integration Points
- **External Dependencies**: `spatial-grouping-attention`, `rotary-spatial-embeddings` packages
- **Core Import Pattern**: Import from these packages, not local implementations
- **Model Integration**: Wrappers in `experiments/common/models.py` for segmentation/detection tasks

## Experiment Framework Structure

### Configuration Management
- **Centralized Configs**: All task-specific configs in `experiments/configs/`
- **Ray Integration**: Configs designed for Ray Train distributed execution
- **DeepSpeed Integration**: Automatic DeepSpeed config generation for optimization

### Multi-Experiment Coordination
- **Simple Experiment Runner**: `experiments/run_experiment.py` - single experiment with dedicated configs (RECOMMENDED)
- **Distributed Training**: `experiments/ray_train.py` - Ray Train with DeepSpeed optimization
- **Comprehensive Evaluation**: `experiments/ray_evaluate.py` - multi-metric evaluation with robustness testing


### Data Pipeline Architecture
```python
# Multi-scale dataset pattern in experiments/common/datasets.py
class ISICDataset(Dataset):
    def __init__(self, multi_scale=False, scales=[256, 128, 64]):
        self.scales = scales if multi_scale else [scales[0]]

    def __getitem__(self, idx):
        if self.multi_scale:
            return [self.transform_to_scale(image, s) for s in self.scales]
        return self.transform(image)
```

### Distributed Training Patterns
```python
# Standard distributed setup in train.py files
def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0
```

### DeepSpeed Integration Framework
- **Automatic Config Creation**: `create_deepspeed_config()` in training scripts
- **ZeRO Stages**: Stage 2 (default), Stage 3 (ViT-Huge scale) with CPU offloading
- **Memory Optimization**: Activation checkpointing, gradient accumulation, FP16

### Model Configuration Pattern
```yaml
# Standard config structure across all experiments
model:
  name: "rat"
  multi_scale: true/false
  spatial_dims: 2  # or 3 for volumes
  feature_dims: 128  # Embedding dimension
  num_blocks: 4      # Transformer layers
  sga_attention_type: "dense"  # or "sparse"

# Training configuration with DeepSpeed support
training:
  batch_size: 4      # Per-GPU batch size
  learning_rate: 0.0001
  grad_clip: 1.0     # Always use gradient clipping
  scheduler: "cosine"

# DeepSpeed integration (optional)
deepspeed:
  zero_stage: 2      # or 3 for maximum memory efficiency
  cpu_offload: false # Enable for Stage 3
```

## Development Workflows

### Local Development
```bash
# Setup with development dependencies
make install-dev
# Run tests with coverage
make test-cov
# Format and lint
make format && make lint
```

### Cluster Deployment
- **Ray Train Integration**: Automatic cluster resource detection and optimization
- **DeepSpeed Support**: Built-in memory optimization for large-scale training
- **Log Management**: Centralized logging in `experiments/results/` directory

### Experiment Execution
```bash
# Simple experiment runner (RECOMMENDED)
python experiments/run_experiment.py --config configs/medical_segmentation.yaml --num-gpus 4

# Individual training with Ray optimization
python experiments/ray_train.py --config configs/medical_segmentation.yaml --num-gpus 4

# Comprehensive evaluation with robustness testing
python experiments/ray_evaluate.py --config configs/medical_segmentation.yaml \
  --checkpoint results/checkpoints/best_model.pth --robustness --num-gpus 2

# Evaluation-only mode (using pretrained checkpoint)
python experiments/run_experiment.py --config configs/medical_segmentation.yaml \
  --evaluation-only --checkpoint results/checkpoints/best_model.pth --num-gpus 2

# Quick testing mode
python experiments/run_experiment.py --config configs/medical_segmentation.yaml --quick --num-gpus 2


```

### Advanced Execution Patterns
```bash
# Distributed training with automatic DeepSpeed optimization
python experiments/ray_train.py --config configs/cluster_rat_multiscale.yaml --num-gpus 8

# Multi-resolution robustness evaluation
python experiments/ray_evaluate.py --config configs/rat_multiscale.yaml \
  --checkpoint results/checkpoints/best_model.pth \
  --robustness --resolutions 128 256 512 1024 --num-gpus 4

# Robustness testing with simple runner
python experiments/run_experiment.py --config configs/rat_multiscale.yaml \
  --evaluation-only --checkpoint results/checkpoints/best_model.pth --robustness --num-gpus 4

# Evaluation-only workflow for pretrained models
python experiments/ray_evaluate.py --config configs/rat_multiscale.yaml \
  --checkpoint pretrained/rat_model.pth --num-gpus 2
```

## Critical Patterns & Conventions

### Memory Management for Multi-Scale
- **Reduced Dimensions**: Use `feature_dims: 128`, `num_blocks: 2` for memory-constrained environments
- **Batch Size Scaling**: Start with `batch_size: 4` for multi-scale experiments
- **Gradient Clipping**: Always use `grad_clip: 1.0` in training configs

### Error Handling Patterns
```python
# Standard try/catch for model imports
try:
    from resolution_aware_transformer import ResolutionAwareTransformer
except ImportError:
    print("Warning: Could not import RAT. Make sure it's installed.")
    ResolutionAwareTransformer = None
```

### Spatial Dimension Handling
- **2D Images**: `spatial_dims: 2`, input shape `[batch, channels, H, W]`
- **3D Volumes**: `spatial_dims: 3`, input shape `[batch, channels, D, H, W]`
- **Spacing Aware**: Use `input_spacing` parameter for medical imaging with known pixel sizes

## Testing & Validation

### Unit Tests
- **Location**: `tests/test_resolution_aware_transformer.py`
- **Coverage Target**: Focus on multi-scale input/output shape consistency
- **Run Command**: `pytest tests/` or `make test`

### Integration Tests
- **Quick Validation**: Use `--quick` flag with `run_experiment.py` for rapid testing
- **Memory Testing**: Automatic batch size optimization handles memory constraints

## File Organization Rules

### Source Code
- **Core Model**: `src/resolution_aware_transformer/` - pure PyTorch implementation
- **Experiments**: `experiments/` - application-specific wrappers and training scripts
- **Common Utilities**: `experiments/common/` - shared datasets, models, metrics, utils

### Configuration Files
- **Unified Configs**: Single YAML configs in `experiments/configs/` for all environments
- **Ray Train Integration**: Configs optimized for Ray Train distributed execution
- **Automatic Optimization**: DeepSpeed and batch size optimization built-in

### Results & Logging
- **TensorBoard Logs**: `experiments/results/tensorboard_logs/`
- **Ray Train Logs**: `experiments/results/ray_logs/` (organized by experiment)
- **Experiment Outputs**: `experiments/results/experiment_logs/`

## Key Debugging Points

### Common Runtime Errors (From Actual Logs)

#### 1. DeepSpeed Argument Conflicts
**Error**: `argparse.ArgumentError: argument --deepspeed: conflicting option string`
```python
# Fix: Check for existing DeepSpeed arguments before adding
if not hasattr(parser, '_option_string_actions') or '--deepspeed' not in parser._option_string_actions:
    args = deepspeed.add_config_arguments(parser)
```

#### 2. Multi-GPU Process Collision
**Error**: Multiple processes starting same experiment simultaneously
**Symptoms**: Duplicate log entries, garbled output in LSF logs
```bash
# Fix: Use RANK-aware initialization
if int(os.environ.get('RANK', 0)) == 0:
    # Only rank 0 process handles logging/checkpointing
    setup_experiment_tracking()
```

#### 3. Dataset API Mismatches
**Error**: `COCODataset.__init__() got an unexpected keyword argument 'augment'`
```python
# Fix: Validate dataset constructor signatures
from experiments.common.datasets import COCODataset
# Check __init__ parameters before passing custom arguments
valid_params = inspect.signature(COCODataset.__init__).parameters
filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
```

#### 4. Pillow Deprecation Warnings
**Error**: `'mode' parameter is deprecated and will be removed in Pillow 13`
```python
# Fix: Remove explicit mode parameter
# OLD: mask = Image.fromarray(mask_array, mode="L")
# NEW: mask = Image.fromarray(mask_array.astype(np.uint8))
```

#### 5. OMP Threading Conflicts
**Warning**: `Setting OMP_NUM_THREADS environment variable for each process to be 1`
```bash
# Fix: Set optimal threading before distributed launch
export OMP_NUM_THREADS=4  # Adjust based on CPU cores per GPU
export MKL_NUM_THREADS=4
```

### Memory Debugging Workflow

#### Stage 1: Quick Memory Test
```bash
# Start with minimal config for memory profiling
python experiments/ray_train.py \
  --config configs/medical_segmentation.yaml \
  --num-gpus 1  # Single GPU testing
```

#### Stage 2: Progressive Scaling
```yaml
# Increment parameters in this order:
# 1. batch_size: 2 → 4 → 8
# 2. feature_dims: 64 → 128 → 256
# 3. num_blocks: 1 → 2 → 4
# 4. multi_scale: false → true
```

#### Stage 3: Multi-GPU Memory Issues
```bash
# Check CUDA memory on each GPU during training
nvidia-smi -l 1 | tee gpu_usage.log
# Look for uneven memory distribution across GPUs
```

### Ray Train Job Debugging

#### Monitor Ray Train Jobs
```bash
# Monitor Ray cluster status
ray status

# Check Ray Train experiment progress
tail -f experiments/results/ray_logs/experiment_*.log

# Monitor GPU usage across Ray workers
ray exec experiments/configs/cluster.yaml 'nvidia-smi'
```

#### Resource Optimization
```bash
# Ray automatically handles resource allocation
# Check allocated resources in Ray dashboard
ray dashboard

# Debug distributed training
echo "Ray worker rank: $RAY_TRAIN_RANK"
echo "Ray world size: $RAY_TRAIN_WORLD_SIZE"
echo "Local GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"
```

### Configuration Debugging Patterns

#### Validate Config Loading
```python
# Always validate config structure before training
def validate_config(config):
    required_keys = ['model', 'training', 'data']
    for key in required_keys:
        assert key in config, f"Missing required config section: {key}"

    # Check model parameters for memory feasibility
    if config['model']['feature_dims'] > 512 and not config.get('deepspeed'):
        print("Warning: Large feature_dims without DeepSpeed may cause OOM")
```

#### Cluster vs Local Config Detection
```python
# Auto-detect execution environment
def get_config_variant(config_path):
    if 'SLURM_JOB_ID' in os.environ or 'LSB_JOBID' in os.environ:
        cluster_config = config_path.replace('.yaml', '_cluster.yaml')
        return cluster_config if Path(cluster_config).exists() else config_path
    return config_path
```

### Common Issues
1. **CUDA OOM**: Reduce `feature_dims`, `num_blocks`, or `batch_size` in configs
2. **Multi-Scale Memory**: Disable `multi_scale: false` for initial testing
3. **Import Errors**: Ensure `spatial-grouping-attention` and `rotary-spatial-embeddings` are installed
4. **Cluster Job Failures**: Check LSF logs in `experiments/results/lsf_logs/`
5. **DeepSpeed Conflicts**: Verify argument parsing order and avoid duplicate argument registration

When modifying the core transformer architecture, always test with both 2D and 3D inputs, and verify multi-scale processing maintains consistent embedding dimensions across resolution levels.
