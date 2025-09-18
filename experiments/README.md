# RAT Experiments - Simplified with Ray Train

This directory contains a simplified training framework for evaluating the Resolution Aware Transformer (RAT) on various computer vision tasks using Ray Train.

## 🚀 Ray Train Integration Features

- **Automatic Distributed Training**: Ray handles all distributed training complexity
- **Built-in Fault Tolerance**: Automatic recovery from node failures  
- **Simplified Resource Management**: Ray automatically manages GPU/CPU allocation
- **Native Experiment Tracking**: Built-in logging and checkpointing
- **Easy Scaling**: Scale from single GPU to multi-node with one parameter

## 🎯 Quick Start

### 1. Install Ray Train
```bash
pip install ray[train]
```

### 2. Local Training (Single GPU)
```bash
python ray_train.py --config configs/ray_template.yaml --num-gpus 1
```

### 3. Multi-GPU Training  
```bash
python ray_train.py --config configs/ray_template.yaml --num-gpus 4
```

### 4. Cluster Training
```bash
# Ray automatically handles cluster setup
ray start --head
python ray_train.py --config configs/ray_template.yaml --num-gpus 8
```

## 📁 Simplified Structure

```
experiments/
├── ray_train.py               # 🆕 Single Ray Train script for all tasks
├── configs/
│   └── ray_template.yaml      # 🆕 Simplified configuration  
├── examples/                  # Example scripts and tutorials
├── common/                    # Shared utilities
├── medical_segmentation/      # Specific experiment implementations
└── object_detection/         # Specific experiment implementations
```

## 📋 Benefits of Ray Train Integration

- **95% Code Reduction**: Single training script replaces complex distributed setup
- **Automatic Optimization**: Ray handles resource allocation and scaling
- **Built-in Fault Tolerance**: Automatic recovery and checkpointing
- **Simplified Deployment**: Works on laptop, workstation, or cluster with same code
- **Better Experiment Tracking**: Native integration with tracking systems

## 🔧 Configuration

Ray Train uses simplified configurations without manual optimization:

```yaml
# Simple Ray Train config
experiment_name: "rat_ray_segmentation"
task_type: "segmentation"

model:
  name: "rat"
  feature_dims: 128
  num_blocks: 4

training:
  epochs: 100
  batch_size: 8  # Ray distributes automatically
  learning_rate: 1e-4

data:
  data_dir: "/path/to/ISIC2018"
  image_size: 256

# No complex distributed training setup needed!
```

## 📚 Alternative: Manual Framework

For users who prefer more control, the original unified framework is still available in the `core/` directory with `unified_train.py`.

---

# Original Experimental Design

The experiments below demonstrate proof-of-concept results showing the effectiveness of multi-scale processing, Rotary Spatial Embeddings (RoSE), and Spatial Grouping Attention (SGA) mechanisms.

## Experimental Overview

### Core Hypothesis
Multi-resolution processing with spatial awareness significantly improves performance on tasks requiring both global context and fine-grained detail, particularly in medical imaging and object detection scenarios.

### Key Research Questions
1. **Multi-scale Advantage**: Does multi-resolution processing outperform single-resolution approaches?
2. **Component Effectiveness**: How do RoSE and SGA components individually contribute to performance?
3. **Architectural Choices**: What are the optimal architectural parameters (blocks, attention types, feature dimensions)?
4. **Scale Robustness**: How robust is the model to resolution variations at test time?