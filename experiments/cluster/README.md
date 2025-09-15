# RAT Cluster Computing Setup

This directory contains scripts and configurations for running Resolution Aware Transformer experiments efficiently on compute clusters with multi-GPU nodes.

## 🚀 Quick Start

### 1. Test Setup (Recommended First)
```bash
# Submit a quick 2-GPU test job
sbatch cluster_scripts/submit_quick_test.slurm
```

### 2. Full Experiments
```bash
# Submit full 8-GPU experiment suite
sbatch cluster_scripts/submit_experiments.slurm
```

### 3. Monitor Progress
```bash
# Check job status
squeue -u $USER

# Monitor TensorBoard logs (on login node)
tensorboard --logdir /shared/results/rat_experiments/tensorboard_logs --port 6006

# Check experiment logs
tail -f /shared/results/rat_experiments/slurm_logs/rat_*.out
```

## 📁 Directory Structure

```
cluster_scripts/
├── submit_experiments.slurm      # Main 8-GPU experiment submission
├── submit_quick_test.slurm       # Quick 2-GPU validation test
└── README.md                     # This file

scripts/
├── setup_cluster.py              # Full cluster setup script
├── update_cluster_configs.py     # Configuration updater
├── setup_isic.py                 # ISIC dataset setup
├── setup_coco.py                 # COCO dataset setup
└── setup_experiments.py          # Original setup script

experiments/
├── train_distributed.py          # Distributed training script
└── configs/                      # Configuration files
```

## 🔧 Cluster-Optimized Features

### **Data Management**
- **Local Storage**: Datasets copied to fast local storage (`/tmp/`) for efficient I/O
- **Network Storage**: Results and checkpoints saved to shared network storage
- **Automatic Cleanup**: Local data cleaned up after job completion

### **Multi-GPU Training**
- **Distributed Training**: Uses PyTorch DistributedDataParallel (DDP)
- **NCCL Backend**: Optimized for multi-GPU communication
- **Mixed Precision**: FP16 training for memory efficiency
- **Gradient Scaling**: Automatic loss scaling for stable training

### **Logging & Monitoring**
- **TensorBoard**: Replaces WandB for cluster-friendly logging
- **Experiment Tracking**: Comprehensive logs and metrics
- **Resource Monitoring**: GPU utilization and memory tracking

### **Configuration Management**
- **Cluster Configs**: Automatically generated cluster-optimized configurations
- **Batch Size Scaling**: Automatic batch size adjustment for multi-GPU
- **Resource Optimization**: Optimal worker counts and memory settings

## 📊 Hardware Requirements

### **Recommended Node Specifications**
- **GPUs**: 8x V100/A100/H100 (minimum 16GB VRAM each)
- **CPU**: 32+ cores
- **RAM**: 256GB+ system memory
- **Storage**: 100GB+ local fast storage (NVMe SSD preferred)
- **Network**: High-speed interconnect for multi-node (if needed)

### **Minimum Requirements**
- **GPUs**: 2x GPUs with 8GB+ VRAM
- **CPU**: 8+ cores
- **RAM**: 32GB+ system memory
- **Storage**: 20GB+ local storage

## 🎯 Experiment Configuration

### **Automatic Optimizations**
- **Batch Size**: Automatically scaled based on number of GPUs
- **Workers**: Set to 2 workers per GPU for optimal I/O
- **Memory**: Pin memory and persistent workers enabled
- **Precision**: Mixed precision training enabled by default

### **Key Configuration Changes**
```yaml
# Original config
training:
  batch_size: 16
  epochs: 100

# Cluster-optimized config  
training:
  batch_size: 2              # 16 / 8 GPUs
  effective_batch_size: 16   # Maintained through gradient accumulation
  epochs: 100
  mixed_precision: true
  gradient_accumulation_steps: 4

cluster:
  num_gpus: 8
  distributed_backend: "nccl"
  find_unused_parameters: true

logging:
  backend: "tensorboard"     # Replaced WandB
  log_dir: "/shared/results/rat_experiments/tensorboard_logs"
```

## 📈 Performance Expectations

### **Speed Improvements**
- **8-GPU vs 1-GPU**: ~6-7x speedup (accounting for communication overhead)
- **Local Storage**: 2-3x faster data loading vs network storage
- **Mixed Precision**: 1.5-2x memory efficiency, slight speed improvement

### **Estimated Runtimes** (8 V100 GPUs)
- **Medical Segmentation**: 2-4 hours per experiment
- **Object Detection**: 4-8 hours per experiment  
- **Ablation Studies**: 1-2 hours per experiment
- **Complete Suite**: 12-16 hours total

## 🔧 Customization

### **Modify for Your Cluster**

1. **Update SLURM Parameters**:
```bash
# Edit cluster_scripts/submit_experiments.slurm
#SBATCH --partition=gpu          # Your GPU partition
#SBATCH --account=your_account   # Your account/allocation
#SBATCH --constraint=v100        # Specific GPU type
```

2. **Update Environment Setup**:
```bash
# Edit environment loading in SLURM scripts
module load cuda/11.8           # Your CUDA module
conda activate your_env         # Your conda environment
```

3. **Update Storage Paths**:
```bash
# Modify paths in scripts for your storage layout
LOCAL_DATA_DIR="/scratch/user/rat_data"     # Your fast local storage
NETWORK_RESULTS_DIR="/projects/user/results" # Your shared storage
```

### **Scale to Different GPU Counts**
```bash
# For 4 GPUs
sed -i 's/gres=gpu:8/gres=gpu:4/g' cluster_scripts/submit_experiments.slurm
sed -i 's/NUM_GPUS=8/NUM_GPUS=4/g' cluster_scripts/submit_experiments.slurm

# For 16 GPUs (multi-node)
# Add --nodes=2 and update distributed training setup
```

## 📋 Monitoring & Debugging

### **Check Job Status**
```bash
# Job queue status
squeue -u $USER

# Detailed job info
scontrol show job <JOB_ID>

# Job efficiency
seff <JOB_ID>
```

### **Monitor Training**
```bash
# Real-time logs
tail -f /shared/results/rat_experiments/slurm_logs/rat_*.out

# TensorBoard (on login node)
tensorboard --logdir /shared/results/rat_experiments/tensorboard_logs

# GPU utilization (if job is running)
ssh <compute_node> nvidia-smi
```

### **Common Issues & Solutions**

1. **Out of Memory**:
   - Reduce batch size in configs
   - Enable gradient checkpointing
   - Use smaller model variants

2. **Slow Data Loading**:
   - Verify data is on local storage
   - Increase number of workers
   - Check I/O bottlenecks

3. **Distributed Training Failures**:
   - Check network connectivity between GPUs
   - Verify NCCL configuration
   - Check for CUDA version compatibility

## 🎯 Optimization Tips

### **Data Loading**
- Always copy datasets to local node storage
- Use multiple workers (2x number of GPUs)
- Enable pin_memory for faster GPU transfers

### **Memory Management**
- Use mixed precision training
- Enable gradient checkpointing for large models
- Monitor memory usage with nvidia-smi

### **Communication**
- Use NCCL backend for multi-GPU communication
- Ensure high-speed interconnect between GPUs
- Monitor communication overhead in logs

## 📊 Results Organization

### **Output Structure**
```
/shared/results/rat_experiments/
├── tensorboard_logs/           # TensorBoard event files
├── slurm_logs/                # SLURM stdout/stderr logs
├── experiment_logs/           # Detailed experiment logs
├── experiment_summary_*.txt   # Job summaries
└── metrics/                   # Exported metrics and plots

/shared/checkpoints/rat/
├── medical_segmentation/      # Model checkpoints by experiment
├── object_detection/
└── ablations/
```

This setup provides efficient, scalable, and monitored execution of RAT experiments on compute clusters while maximizing resource utilization and minimizing I/O bottlenecks.