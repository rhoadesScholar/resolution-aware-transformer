# Experiment Results Organization System

This directory contains an improved experiment tracking and results organization system for the Resolution Aware Transformer project.

## 🎯 Problem Solved

The original results structure was disorganized with:
- Deeply nested Ray Train directories (`rat_medical_segmentation_isic2018/rat_medical_segmentation_isic2018/TorchTrainer_094cf_00000_0_2025-09-26_15-43-12/`)
- Scattered log files across different locations
- No clear distinction between successful and failed experiments
- Difficult to correlate system logs with experiment failures
- No centralized experiment tracking

## 🏗️ New Directory Structure

```
results/
├── experiments/
│   ├── active/           # Currently running experiments
│   ├── completed/        # Successfully completed experiments  
│   ├── failed/          # Failed experiments with error analysis
│   └── archived/        # Older experiments moved for cleanup
├── logs/
│   ├── system/          # LSF, SLURM, Ray cluster logs
│   ├── training/        # Training logs by experiment
│   └── evaluation/      # Evaluation logs by experiment
├── checkpoints/
│   ├── by_experiment/   # Checkpoints organized by experiment
│   └── best_models/     # Best performing models across experiments
├── tensorboard/         # TensorBoard logs organized by experiment
├── analysis/            # Analysis reports and comparisons
└── experiment_registry.json  # Central experiment tracking
```

## 🚀 Quick Start

### 1. Organize Existing Results

```bash
# Preview what will be organized (safe, no changes)
python organize_results.py --dry-run

# Create backup and organize (recommended)
python organize_results.py --backup

# Just organize (if you're confident)
python organize_results.py
```

### 2. Run New Experiments (Automatic Organization)

The updated experiment runners automatically use the new organization:

```bash
# Medical segmentation experiment
python run_experiment.py --config configs/medical_segmentation.yaml --num-gpus 8

# Quick test run
python run_experiment.py --config configs/medical_segmentation.yaml --quick --num-gpus 4

# Evaluation only
python run_experiment.py --config configs/medical_segmentation.yaml --evaluation-only --checkpoint results/checkpoints/best_models/best_model.pth
```

### 3. Experiment Tracking Commands

```bash
# View experiment summary
python -m common.experiment_tracker summary

# Generate detailed report
python -m common.experiment_tracker report

# Clean up old experiments (archive experiments older than 30 days)
python -m common.experiment_tracker cleanup --days 30

# Delete old experiments (permanently remove instead of archiving)
python -m common.experiment_tracker cleanup --days 30 --delete
```

## 📊 Experiment Tracking Features

### Automatic Registration
- Each experiment gets a unique ID: `{experiment_name}_{timestamp}`
- Metadata tracking: config, start/end times, status, resources used
- Organized directory structure created automatically

### Status Management
- **Active**: Currently running experiments
- **Completed**: Successfully finished experiments with all results
- **Failed**: Failed experiments with error analysis
- **Archived**: Old experiments moved for cleanup

### Error Analysis
- Automatic error capture and categorization
- Correlation between system logs and experiment failures
- Error types tracked (OOM, config errors, data errors, etc.)

## 🔧 Integration with Existing Tools

### Ray Train Integration
The new system works seamlessly with existing Ray Train workflows:

```python
from common.experiment_tracker import ExperimentTracker

# In your training script
tracker = ExperimentTracker()
experiment_id = tracker.register_experiment(
    experiment_name="medical_segmentation",
    config_path="configs/medical_segmentation.yaml",
    task_type="segmentation",
    num_gpus=8
)

# Training happens here...

# Update status on completion/failure
tracker.update_experiment_status(experiment_id, "completed", results_info)
```

### LSF/SLURM Integration
System logs are automatically organized:

```bash
# LSF logs go to results/logs/system/
# Correlated with experiment logs in results/logs/training/{experiment_id}/
```

## 📈 Benefits

### 1. **Easier Debugging**
- All logs for an experiment in one place
- Clear error categorization and analysis
- System logs correlated with experiment failures

### 2. **Better Resource Management**
- Track GPU usage and training times across experiments
- Identify resource-intensive configurations
- Automatic cleanup of old experiments

### 3. **Experiment Comparison**
- Centralized registry makes it easy to compare experiments
- Best model tracking across all experiments
- Analysis reports for experiment comparison

### 4. **Reduced Storage Bloat**
- Automatic archival of old experiments
- Configurable cleanup policies
- Backup integration for safety

## 🛠️ Advanced Usage

### Custom Results Directory
```bash
# Use custom results directory
python organize_results.py --results-dir /path/to/custom/results
python -m common.experiment_tracker summary --results-dir /path/to/custom/results
```

### Programmatic Access
```python
from common.experiment_tracker import ExperimentTracker

tracker = ExperimentTracker("/path/to/results")

# Get all completed experiments
completed = [exp for exp in tracker.registry["experiments"].values() 
            if exp["status"] == "completed"]

# Find best performing experiment
best_experiment = max(completed, 
                     key=lambda x: x.get("additional_info", {}).get("dice_score", 0))

# Generate custom analysis
report_path = tracker.generate_experiment_report()
```

### Backup and Recovery
```bash
# Create backup before major changes
python organize_results.py --backup

# Restore from backup if needed
mv results_backup_20250926_153045 results
```

## 🔍 Troubleshooting

### Migration Issues
If you encounter issues during organization:

```bash
# Run dry-run first to see what would happen
python organize_results.py --dry-run

# Always create backup for safety
python organize_results.py --backup

# Check the experiment tracker log
tail -f results/experiment_tracker.log
```

### Ray Train Directory Structure
The new system handles Ray Train's complex directory structure by:
1. Extracting meaningful experiment information
2. Flattening nested directories
3. Preserving all original files in organized locations
4. Creating human-readable directory names

### Log Correlation
To correlate system logs with experiment failures:
1. Check `results/logs/system/` for LSF/SLURM logs
2. Check `results/experiments/failed/{experiment_id}/logs/` for experiment logs
3. Use `experiment_registry.json` to map experiment IDs to timestamps

## 📚 API Reference

### ExperimentTracker Class

```python
class ExperimentTracker:
    def __init__(self, base_results_dir: str = "results")
    def register_experiment(self, experiment_name: str, config_path: str, 
                          task_type: str, num_gpus: int = 1) -> str
    def update_experiment_status(self, experiment_id: str, status: str, 
                               additional_info: Dict = None)
    def organize_existing_results(self, source_dir: Path = None)
    def get_experiment_summary(self) -> Dict
    def generate_experiment_report(self, output_path: Path = None) -> Path
    def cleanup_old_experiments(self, days_old: int = 30, archive: bool = True)
```

### Command Line Tools

```bash
# Experiment tracker CLI
python -m common.experiment_tracker [organize|summary|report|cleanup]

# Results organization tool  
python organize_results.py [--dry-run] [--backup] [--results-dir DIR]
```

## 🤝 Contributing

When adding new experiment types or modifying the tracking system:

1. Update the `ExperimentTracker` class if needed
2. Ensure new experiment runners integrate with the tracking system
3. Add tests for new functionality
4. Update this documentation

## 📝 Changelog

- **v1.0** (2024-09-26): Initial implementation with automatic organization and tracking
- Better Ray Train integration
- LSF/SLURM log correlation
- Automatic cleanup and archival
- Comprehensive reporting system