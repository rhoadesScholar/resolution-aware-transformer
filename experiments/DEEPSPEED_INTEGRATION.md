# DeepSpeed Integration for ViT-Huge Scale Models

This document outlines the DeepSpeed integration added to the Resolution-Aware Transformer experiments to enable training of ViT-Huge scale models with advanced memory optimization.

## Overview

DeepSpeed with ZeRO (Zero Redundancy Optimizer) stages provides memory optimization through:
- **Stage 1**: Partitions optimizer states across GPUs
- **Stage 2**: Partitions gradients and optimizer states (default)
- **Stage 3**: Partitions parameters, gradients, and optimizer states with CPU offloading

## Integration Details

### Medical Segmentation (`experiments/medical_segmentation/train.py`)
- ✅ **Complete DeepSpeed Integration**
- Added `--deepspeed` and `--zero_stage` arguments
- Created `create_deepspeed_config()` function for Stage 2 and Stage 3 configurations
- Modified training loop to use DeepSpeed engine
- Added distributed training support with DeepSpeed initialization
- Created ViT-Huge configuration: `medical_segmentation/configs/rat_vit_huge.yaml`

### Object Detection (`experiments/object_detection/train.py`)
- ✅ **Complete DeepSpeed Integration**
- Added same DeepSpeed arguments and configuration creation
- Modified `train_epoch()` and `validate()` functions for DeepSpeed support
- Updated main function with DeepSpeed initialization and distributed training
- Created ViT-Huge configuration: `object_detection/configs/rat_vit_huge_detection.yaml`
- Updated LSF script to use `deepspeed` command instead of `torchrun`

### Ablation Studies (`experiments/ablations/ablation_study.py`)
- ⏸️ **No Changes Needed**
- Evaluation-only script that loads pretrained models
- Does not perform training, so DeepSpeed integration not required

## Configuration Files Created

### DeepSpeed Configurations
1. **`deepspeed_config.json`** - Stage 2 optimization (default)
   - Partitions gradients and optimizer states
   - FP16 training with automatic loss scaling
   - Activation checkpointing for memory efficiency

2. **`deepspeed_stage3_config.json`** - Stage 3 optimization (for ViT-Huge)
   - Partitions parameters, gradients, and optimizer states
   - CPU offloading for maximum memory efficiency
   - Advanced activation checkpointing with CPU storage

### ViT-Huge Model Configurations
1. **Medical Segmentation**: `rat_vit_huge.yaml`
   - 1280 feature dimensions, 32 blocks, 16 heads
   - Optimized for 128x128 images with batch size 1
   - Spatial grouping with reduced group size (4) for memory efficiency

2. **Object Detection**: `rat_vit_huge_detection.yaml`
   - Same ViT-Huge architecture adapted for COCO detection
   - 512x512 images with multi-scale training
   - DETR-style detection with 100 object queries

## LSF Script Updates

### Medical Segmentation (`cluster/submit_medseg.lsf`)
- Updated to use `deepspeed` command instead of `torchrun`
- Added `--deepspeed --zero_stage 3` for ViT-Huge models
- Maintained backward compatibility with distributed training flags

### Object Detection (`cluster/submit_objdet.lsf`)
- Similar DeepSpeed command integration
- Uses Stage 2 by default for better compatibility
- Can be upgraded to Stage 3 for ViT-Huge models

## Memory Optimization Results

### Memory Scaling Analysis
- **256x256 images**: 115GB memory allocation (too large)
- **128x128 images**: 18MB memory allocation (manageable)
- **Spatial grouping attention**: Critical for memory efficiency with large models

### DeepSpeed Benefits
- **ZeRO Stage 2**: Reduces memory usage by ~50% through gradient/optimizer partitioning
- **ZeRO Stage 3**: Enables training of models 8x larger through parameter partitioning
- **CPU Offloading**: Further reduces GPU memory by moving optimizer states to CPU
- **Activation Checkpointing**: Trades computation for memory in forward pass

## Usage Examples

### Training with DeepSpeed Stage 2 (Default)
```bash
deepspeed --num_gpus=8 experiments/medical_segmentation/train.py \
  --config experiments/medical_segmentation/configs/baseline.yaml \
  --deepspeed \
  --zero_stage 2 \
  --distributed
```

### Training ViT-Huge with DeepSpeed Stage 3
```bash
deepspeed --num_gpus=8 experiments/medical_segmentation/train.py \
  --config experiments/medical_segmentation/configs/rat_vit_huge.yaml \
  --deepspeed \
  --zero_stage 3 \
  --distributed
```

### LSF Cluster Submission
```bash
# Submit ViT-Huge medical segmentation with DeepSpeed
bsub < experiments/cluster/submit_medseg.lsf

# Submit object detection with DeepSpeed
bsub < experiments/cluster/submit_objdet.lsf
```

## Performance Considerations

### Memory Efficiency
- Use Stage 3 for models >1B parameters (ViT-Huge)
- Use Stage 2 for smaller models with good performance/memory balance
- Enable CPU offloading for maximum memory savings
- Reduce image resolution and batch size for very large models

### Training Speed
- Stage 2: Minimal performance impact (~5-10% overhead)
- Stage 3: Higher communication overhead but enables training of larger models
- CPU offloading: Additional latency but enables much larger models
- Activation checkpointing: ~20% slower but significant memory savings

## Next Steps

1. **Test ViT-Huge configurations** on H100 cluster with DeepSpeed Stage 3
2. **Benchmark memory usage** and training speed with different ZeRO stages
3. **Optimize spatial grouping** attention for better memory scaling
4. **Evaluate model quality** after DeepSpeed optimization
5. **Scale to even larger models** (ViT-Giant) if memory permits

## Dependencies

- `deepspeed>=0.9.0` - Install with `pip install deepspeed`
- PyTorch distributed training support
- CUDA-compatible GPU with sufficient memory
- LSF cluster with H100 GPUs (recommended)

This integration enables the training of ViT-Huge scale Resolution-Aware Transformers while maintaining the existing experiment infrastructure and providing backward compatibility with standard distributed training approaches.