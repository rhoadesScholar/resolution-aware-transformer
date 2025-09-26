# Experiment Logs Directory

This directory contains application-level log files from RAT experiments.

## Log Organization

- **Medical Segmentation**: `medical_segmentation_train_YYYYMMDD_HHMMSS.log`
- **Object Detection**: `object_detection_train_YYYYMMDD_HHMMSS.log`
- **Ablation Studies**: `ablation_study_YYYYMMDD_HHMMSS.log`
- **Robustness Tests**: `robustness_test_YYYYMMDD_HHMMSS.log`

## Log Types

### Application Logs (this directory)
- Training progress and metrics
- Model configuration details
- Error messages and debugging info
- Experiment tracking information

### LSF Job Logs (`../lsf_logs/`)
- Cluster job stdout/stderr
- Resource allocation info
- Job scheduling details

### TensorBoard Logs (`../tensorboard_logs/`)
- Training metrics for visualization
- Model graphs and histograms
- Scalars and images

## Viewing Logs

```bash
# View latest training log
tail -f medical_segmentation_train_*.log

# View all training logs
ls -la medical_segmentation_train_*.log

# Search for errors
grep -i error *.log
```