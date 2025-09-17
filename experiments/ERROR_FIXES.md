# Error Fixes Applied to RAT Codebase

This document summarizes the fixes applied to resolve the runtime errors found in the LSF logs.

## 1. DeepSpeed Argument Conflict (Fixed)

**Error**: `argparse.ArgumentError: argument --deepspeed: conflicting option string`

**Files Fixed**:
- `experiments/medical_segmentation/train.py`
- `experiments/object_detection/train.py`

**Fix Applied**:
```python
# Before
if DEEPSPEED_AVAILABLE:
    args = deepspeed.add_config_arguments(parser)

# After  
if DEEPSPEED_AVAILABLE and deepspeed is not None:
    if not hasattr(parser, '_option_string_actions') or '--deepspeed' not in parser._option_string_actions:
        parser = deepspeed.add_config_arguments(parser)
```

**Root Cause**: Multiple calls to `deepspeed.add_config_arguments()` were trying to add the same `--deepspeed` argument to the argument parser.

## 2. Pillow Deprecation Warnings (Fixed)

**Error**: `'mode' parameter is deprecated and will be removed in Pillow 13 (2026-10-15)`

**File Fixed**: `experiments/scripts/setup_isic.py`

**Fix Applied**:
```python
# Before
mask = Image.fromarray(mask_array, mode="L")
mask = Image.fromarray(np.zeros((256, 256), dtype=np.uint8), mode="L")

# After
mask = Image.fromarray(mask_array.astype(np.uint8))
mask = Image.fromarray(np.zeros((256, 256), dtype=np.uint8))
```

**Root Cause**: Pillow deprecated the explicit `mode` parameter in `Image.fromarray()`.

## 3. Dataset API Mismatch (Fixed)

**Error**: `COCODataset.__init__() got an unexpected keyword argument 'augment'`

**File Fixed**: `experiments/common/datasets.py`

**Fix Applied**:
```python
# Added missing parameters to COCODataset constructor
def __init__(
    self,
    data_dir: str,
    split: str = "train",
    image_size: Union[int, Tuple[int, int]] = 512,
    multi_scale: bool = False,
    scales: List[int] = [512, 256, 128],
    transform: Optional[Callable] = None,
    augment: bool = False,      # Added
    debug: bool = False,        # Added
):
```

**Root Cause**: The training script was passing `augment` and `debug` parameters that weren't defined in the dataset constructor.

## 4. Multi-GPU Process Collision (Fixed)

**Error**: Multiple processes starting same experiment simultaneously causing duplicate log entries

**File Fixed**: `experiments/ablations/ablation_study.py`

**Fix Applied**:
```python
# Added rank-aware initialization
rank = int(os.environ.get('RANK', 0))
if rank == 0:
    # Only rank 0 process handles logging/checkpointing
    logger = setup_logging(str(log_dir), "ablation_study")
    tracker = ExperimentTracker("ablation_study", str(output_dir))
else:
    logger = None
    tracker = None

# Added conditional logging throughout
if logger:
    logger.info("Message")
if tracker:
    tracker.log_metric("key", value)
```

**Root Cause**: All distributed training processes were trying to initialize logging and experiment tracking simultaneously.

## 5. OMP Threading Conflicts (Addressed)

**Warning**: `Setting OMP_NUM_THREADS environment variable for each process to be 1`

**Solution Created**: 
- `experiments/scripts/setup_environment.sh` - Shell script to set optimal threading
- `experiments/scripts/fix_common_issues.py` - Python script to detect and fix common issues

**Optimal Settings**:
```bash
# Automatically calculated based on CPU cores and GPU count
export OMP_NUM_THREADS=<cores_per_gpu>
export MKL_NUM_THREADS=<cores_per_gpu>
export NUMEXPR_NUM_THREADS=<cores_per_gpu>
```

**Root Cause**: Default PyTorch distributed training sets OMP_NUM_THREADS=1, which is suboptimal for CPU-bound operations.

## 6. DeepSpeed Import Safety (Fixed)

**Error**: Potential `deepspeed` undefined errors when DeepSpeed not available

**Files Fixed**:
- `experiments/medical_segmentation/train.py`
- `experiments/object_detection/train.py`

**Fix Applied**:
```python
# Before
try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False

# After
try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    deepspeed = None
    DEEPSPEED_AVAILABLE = False
```

**Root Cause**: When DeepSpeed import failed, the `deepspeed` variable was undefined, causing runtime errors.

## Usage Instructions

### Quick Fix Script
```bash
# Run before any experiments to check and fix common issues
python3 experiments/scripts/fix_common_issues.py
```

### Environment Setup
```bash
# Source this script to set optimal environment variables
source experiments/scripts/setup_environment.sh

# Then run your experiments
python experiments/medical_segmentation/train.py --config configs/rat_multiscale.yaml
```

### Manual Environment Setup
```bash
# Set optimal threading (adjust based on your hardware)
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

# CUDA optimizations
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_LAUNCH_BLOCKING="1"  # For debugging
```

## Verification

To verify the fixes work:

1. **Check DeepSpeed Integration**:
   ```bash
   python experiments/medical_segmentation/train.py --config configs/rat_multiscale.yaml --deepspeed --debug
   ```

2. **Check Dataset Loading**:
   ```bash
   python experiments/object_detection/train.py --config configs/single_scale.yaml --debug
   ```

3. **Check Multi-GPU Behavior**:
   ```bash
   torchrun --nproc_per_node=2 experiments/ablations/ablation_study.py --config configs/ablation.yaml --quick
   ```

4. **Check Pillow Fixes**:
   ```bash
   python experiments/scripts/setup_isic.py --output_dir /tmp/test_isic --num_samples 10
   ```

## Notes

- All fixes maintain backward compatibility
- The threading optimizations may need adjustment based on specific hardware configurations
- DeepSpeed fixes ensure graceful fallback when DeepSpeed is not available
- Multi-GPU fixes prevent log pollution while maintaining functionality

## Next Steps

1. Test the fixes on a small subset of data
2. Monitor logs for any remaining issues
3. Run full experiments with the optimized environment settings
4. Adjust threading parameters based on performance monitoring