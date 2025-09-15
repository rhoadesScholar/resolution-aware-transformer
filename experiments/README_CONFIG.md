# RAT Experiments Configuration System

This directory contains a comprehensive configuration system for running Resolution Aware Transformer (RAT) experiments on LSF clusters.

## Quick Start

1. **Configure your setup:**
   ```bash
   # Edit the configuration file for your cluster
   nano experiments/.config
   ```

2. **Generate LSF scripts:**
   ```bash
   ./deploy_lsf.sh generate
   ```

3. **Run quick validation:**
   ```bash
   ./deploy_lsf.sh quick
   ```

4. **Run full experiments:**
   ```bash
   ./deploy_lsf.sh full
   ```

## Configuration File: `experiments/.config`

The configuration file uses INI format with the following sections:

### `[cluster]` - LSF Resource Allocation
- `queue`: LSF queue name (default: gpu)
- `walltime_hours`: Maximum job duration for full experiments (default: 48)
- `walltime_hours_quick`: Duration for quick test (default: 2)
- `num_gpus_full/num_gpus_quick`: GPU allocation (default: 8/2)
- `gpu_mode`: GPU allocation mode (default: exclusive_process)
- `cpus_per_gpu`: CPU cores per GPU (default: 4)
- `memory_mb_per_gpu`: Memory in MB per GPU (default: 16000)

### `[paths]` - Directory Configuration
- `repo_root`: Repository root directory
- `data_dir`: Data storage directory (default: repo/data)
- `results_dir`: Results storage directory (default: repo/results)
- `checkpoints_dir`: Model checkpoints directory (default: repo/checkpoints)
- `use_local_storage`: Use local temporary storage for data (default: true)
- `local_temp_dir`: Local temporary directory (default: /tmp/rat_data)

### `[datasets]` - Dataset Configuration
- `isic_source_dir`: Source directory for ISIC dataset (empty = use samples)
- `coco_source_dir`: Source directory for COCO dataset (empty = use samples)
- `use_sample_data`: Use sample datasets for testing (default: true)
- `isic_sample_size/coco_sample_size`: Sample dataset sizes (default: 100/50)

### `[training]` - Training Parameters
- `mixed_precision`: Enable mixed precision training (default: true)
- `distributed_backend`: PyTorch distributed backend (default: nccl)
- `master_port_base`: Base port for distributed training (default: 12355)

### `[experiments]` - Experiment Selection
- `medical_segmentation/object_detection/ablation_studies`: Enable experiments (default: true/true/true)
- `robustness_tests`: Enable robustness testing (default: false)
- `*_epochs`: Number of epochs for each experiment type

### `[environment]` - Environment Setup
- `conda_env`: Conda environment name (optional)
- `modules_to_load`: Comma-separated list of modules to load (optional)
- `job_name_prefix`: Prefix for LSF job names (default: rat)

### `[notifications]` - Job Notifications
- `email`: Email address for job notifications (optional)
- `notification_events`: LSF notification events (default: END,FAIL)

## Tools and Scripts

### `config_manager.py`
Configuration management tool with validation and utilities:
```bash
# Validate configuration
python3 experiments/config_manager.py --validate

# Show current configuration
python3 experiments/config_manager.py --dump

# Create configured directories
python3 experiments/config_manager.py --create-dirs

# Show LSF job configuration
python3 experiments/config_manager.py --lsf-config [--quick]
```

### `generate_lsf_script.py`
Generates LSF job scripts from configuration:
```bash
# Generate full experiments script
python3 experiments/generate_lsf_script.py

# Generate quick test script
python3 experiments/generate_lsf_script.py --quick

# Generate to specific file
python3 experiments/generate_lsf_script.py --output my_job.lsf
```

### `deploy_lsf.sh`
Main deployment script with multiple commands:
```bash
# Show current configuration
./deploy_lsf.sh config

# Generate all LSF scripts
./deploy_lsf.sh generate

# Submit quick test job
./deploy_lsf.sh quick

# Submit full experiments
./deploy_lsf.sh full

# Check job status
./deploy_lsf.sh status
```

## Cluster Optimization Features

- **Local Data Storage**: Automatically copies/creates datasets in local temporary storage for faster I/O
- **Distributed Training**: Full multi-GPU support with PyTorch DDP and NCCL backend
- **Resource Management**: Automatic CPU, memory, and GPU allocation based on configuration
- **TensorBoard Logging**: Centralized experiment tracking and visualization
- **Checkpoint Management**: Automatic model checkpoint saving and organization
- **Mixed Precision**: Optional mixed precision training for improved performance

## Example Workflows

### Development/Testing Workflow
```bash
# 1. Configure for your cluster
nano experiments/.config

# 2. Generate scripts
./deploy_lsf.sh generate

# 3. Validate with quick test
./deploy_lsf.sh quick

# 4. Monitor quick test
bjobs
./deploy_lsf.sh status

# 5. Run full experiments when ready
./deploy_lsf.sh full
```

### Production Workflow
```bash
# 1. Set production datasets in .config
# 2. Disable sample data: use_sample_data = false
# 3. Set source directories: isic_source_dir = /shared/datasets/ISIC2018
# 4. Generate and submit
./deploy_lsf.sh generate
./deploy_lsf.sh full
```

## Monitoring and Results

- **Job logs**: `results/lsf_logs/rat_*_<JOBID>.{out,err}`
- **TensorBoard**: `tensorboard --logdir results/tensorboard_logs`
- **Experiment summaries**: `results/experiment_summary_<JOBID>.txt`
- **Model checkpoints**: `checkpoints/`

## Customization

The configuration system is designed to be easily customizable:

1. **Add new experiments**: Update `[experiments]` section and modify training scripts
2. **Change resource allocation**: Adjust `[cluster]` section for your hardware
3. **Add environment modules**: Set `modules_to_load` in `[environment]` section
4. **Custom notifications**: Configure email/Slack in `[notifications]` section

## Troubleshooting

- **Configuration errors**: Run `python3 experiments/config_manager.py --validate`
- **Path issues**: Check `repo_root` setting and ensure all paths are accessible
- **LSF errors**: Verify queue names and resource limits with your cluster admin
- **Permission issues**: Ensure write access to configured directories

For more details, see individual script help:
```bash
python3 experiments/config_manager.py --help
python3 experiments/generate_lsf_script.py --help
./deploy_lsf.sh help
```