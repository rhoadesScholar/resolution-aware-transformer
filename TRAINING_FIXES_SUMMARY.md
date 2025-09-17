# Training Error Fixes Summary

## Issues Identified and Fixed

### 1. COCO Dataset Annotation Path Error (CRITICAL)
**Issue**: The COCO dataset was constructing annotation filenames incorrectly, resulting in paths like:
```
/tmp/rat_data/rat_data_146605628/annotations/instances_train20172017.json
```
Instead of the correct:
```
/tmp/rat_data/rat_data_146605628/annotations/instances_train2017.json
```

**Root Cause**: The `_load_annotations()` method in `COCODataset` was appending "2017" to splits that already contained "2017" (e.g., "train2017").

**Fix Applied**: Modified the annotation file path construction in `experiments/common/datasets.py`:
```python
# Before:
ann_file = self.data_dir / f"annotations/instances_{self.split}2017.json"

# After:
ann_file = self.data_dir / f"annotations/instances_{self.split}.json"
```

**Impact**: This was causing all object detection experiments to fail immediately upon dataset loading.

### 2. Ablation Study Logger Null Reference Error (CRITICAL)
**Issue**: The ablation study was attempting to use `logger.info()` on non-rank-0 processes where `logger = None`, causing:
```
AttributeError: 'NoneType' object has no attribute 'info'
```

**Root Cause**: Distributed training setup only created a logger on rank 0, but the code didn't check for `None` before using logger methods.

**Fixes Applied**: Added null checks throughout `experiments/ablations/ablation_study.py`:
```python
# Before:
logger.info(f"Running experiment: {exp_config['name']}")

# After:
if logger:
    logger.info(f"Running experiment: {exp_config['name']}")
```

**Impact**: This was causing all ablation study experiments to crash immediately on multi-GPU runs.

### 3. Distributed Training Setup Issues (PREVENTIVE)
**Issue**: Missing error handling around distributed process group initialization.

**Fix Applied**: Added try-catch blocks around `dist.init_process_group()` calls to provide better error messages and prevent silent failures.

### 4. Missing Python Module Structure (PREVENTIVE)
**Issue**: Missing `__init__.py` files in experiment subdirectories could cause import issues.

**Fix Applied**: Added `__init__.py` files to:
- `experiments/common/`
- `experiments/medical_segmentation/`
- `experiments/object_detection/`
- `experiments/ablations/`

### 5. Configuration File Issues (PREVENTIVE)
**Issue**: The ablation configuration file was missing the required `training` section.

**Fix Applied**: Added complete training configuration to `experiments/ablations/configs/ablation.yaml`:
```yaml
training:
  num_epochs: 10
  batch_size: 48
  optimizer:
    name: "adamw"
    lr: 1e-4
    weight_decay: 0.01
  scheduler:
    name: "cosine"
    min_lr: 1e-6
  mixed_precision: true
  gradient_clip: 1.0
```

## Additional Preventive Fixes Created

### Training Issues Detection Script
Created `experiments/scripts/fix_training_issues.py` to:
- Automatically detect and fix common training issues
- Validate configuration files for completeness
- Check for import path issues
- Identify potentially problematic settings (large batch sizes, memory-intensive models)

## Configuration Warnings Identified

### High Batch Sizes
- `baseline.yaml`: batch_size=80 may cause OOM on some GPUs
- **Recommendation**: Consider reducing to 32-48 for initial testing

### Missing Data Paths
- Some configs reference non-existent data directories
- **Recommendation**: Update data paths or use symbolic links to actual datasets

### Large Model Configurations
- Models with feature_dims > 512 or num_blocks > 8 may require DeepSpeed
- **Recommendation**: Enable DeepSpeed for memory-intensive configurations

## Testing Recommendations

1. **Re-run Failed Experiments**: The primary issues (COCO path and logger) have been fixed
2. **Test with Small Datasets**: Use debug mode or small subsets to validate fixes
3. **Monitor GPU Memory**: Check for OOM issues with large batch sizes
4. **Verify Distributed Training**: Test multi-GPU setups with proper rank handling

## Files Modified

1. `experiments/common/datasets.py` - Fixed COCO annotation path
2. `experiments/ablations/ablation_study.py` - Fixed logger null references
3. `experiments/object_detection/train.py` - Added distributed training error handling
4. `experiments/ablations/configs/ablation.yaml` - Added training section
5. Added `__init__.py` files to all experiment subdirectories
6. Created `experiments/scripts/fix_training_issues.py` for future issue detection

All fixes are backward compatible and should not affect existing working configurations.