# LSF Deployment Integration with Fix Scripts

## Overview

The RAT experiment deployment now includes automatic integration of the fix scripts to ensure optimal performance and early error detection.

## Integration Strategy

### 1. Pre-Deployment Checks (`fix_common_issues.py`)

**When**: Run BEFORE submitting LSF jobs  
**Where**: On the submission host via `deploy_lsf.sh`  
**Purpose**: 
- Validate dependencies are installed
- Check file permissions and paths
- Detect common configuration issues
- Provide early feedback before wasting cluster time

### 2. Runtime Environment Setup (`setup_environment.sh`)

**When**: Run INSIDE each LSF job at startup  
**Where**: On the compute node before training begins  
**Purpose**:
- Set optimal threading for the specific hardware
- Configure CUDA memory optimizations
- Set distributed training environment variables

## Updated Deployment Workflow

### Quick Test
```bash
# 1. Automatic pre-checks and fixes
./deploy_lsf.sh quick

# This now automatically:
# - Runs fix_common_issues.py
# - Prompts if issues are found
# - Submits job with environment setup
```

### Full Experiments
```bash
# 1. Automatic pre-checks and fixes
./deploy_lsf.sh full

# This now automatically:
# - Runs fix_common_issues.py
# - Prompts if issues are found
# - Submits jobs with environment setup
```

## Environment Variables Set

### Pre-Deployment (fix_common_issues.py)
- Validates PyTorch installation
- Checks for required packages
- Tests CUDA availability

### Runtime (setup_environment.sh)
```bash
# Threading (calculated per-node)
export OMP_NUM_THREADS=<optimal_value>
export MKL_NUM_THREADS=<optimal_value>
export NUMEXPR_NUM_THREADS=<optimal_value>

# CUDA optimizations
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_LAUNCH_BLOCKING="1"

# Distributed training
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
```

## LSF Scripts Updated

All LSF scripts now include environment setup:
- `submit_quick_test.lsf`
- `submit_medseg.lsf`
- `submit_objdet.lsf` 
- `submit_ablation.lsf`

## Manual Usage

### If you want to run checks manually:
```bash
# Check and fix issues
python3 scripts/fix_common_issues.py

# Set environment (in your current shell)
source scripts/setup_environment.sh

# Then run experiments
python experiments/medical_segmentation/train.py --config configs/rat_multiscale.yaml
```

### If you want to skip pre-checks:
```bash
# Edit deploy_lsf.sh and comment out the fix_common_issues.py call
# Or run bsub directly:
bsub < cluster/submit_quick_test.lsf
```

## Benefits

1. **Early Error Detection**: Issues caught before cluster submission
2. **Optimal Performance**: Environment tuned for each compute node
3. **Reduced Debugging**: Common issues automatically resolved
4. **Consistent Environment**: Same setup across all jobs
5. **Resource Efficiency**: Prevents failed jobs due to configuration issues

## Troubleshooting

### If pre-checks fail:
- Review the output from `fix_common_issues.py`
- Install missing dependencies: `pip install <package>`
- Check file permissions and paths
- Run checks manually to debug

### If environment setup fails in LSF job:
- Check the job output logs for setup_environment.sh output
- Verify the script is executable: `chmod +x scripts/setup_environment.sh`
- Check if the script path is correct in the LSF file

### To disable automatic checks:
- Comment out the `fix_common_issues.py` calls in `deploy_lsf.sh`
- Or modify the LSF scripts to skip `setup_environment.sh`

## Files Modified

### Main deployment:
- `experiments/deploy_lsf.sh` - Added pre-deployment checks

### LSF job scripts:
- `experiments/cluster/submit_quick_test.lsf`
- `experiments/cluster/submit_medseg.lsf`
- `experiments/cluster/submit_objdet.lsf`
- `experiments/cluster/submit_ablation.lsf`

### Fix scripts:
- `experiments/scripts/fix_common_issues.py` - Pre-deployment validation
- `experiments/scripts/setup_environment.sh` - Runtime environment setup

All changes are backward compatible and can be disabled if needed.