# Resolution Aware Transformer: Experiments

This directory contains the experimental validation of the Resolution Aware Transformer (RAT) architecture. Our experiments are designed to demonstrate proof-of-concept results showing the effectiveness of multi-scale processing, Rotary Spatial Embeddings (RoSE), and Spatial Grouping Attention (SGA) mechanisms.

---

## Experimental Overview

### Core Hypothesis
Multi-resolution processing with spatial awareness significantly improves performance on tasks requiring both global context and fine-grained detail, particularly in microscopy, medical imaging, and geospatial scenarios.

### Key Research Questions
1. **Multi-scale Advantage**: Does multi-resolution, multi-FOV processing outperform single-resolution, single-view approaches?
2. **Scale Robustness**: How robust is the model to resolution variations at test time?

## Experiment Structure

### Phase 1: Core Performance Validation

#### 1.1 Medical Image Segmentation
- **Dataset**: ISIC 2018 Skin Lesion Segmentation Challenge
- **Task**: Binary segmentation of melanoma lesions
- **Baselines**: U-Net, Swin-UNet, TransUNet
- **Multi-scale Setup**: Single (256x256), Dual (256x256 + 128x128), Triple (256x256 + 128x128 + 64x64)
- **Metrics**: Dice coefficient, IoU, sensitivity, specificity
- **Rationale**: Medical imaging is a primary target domain where multi-scale features are crucial for accurate boundary delineation

#### 1.2 Multi-Scale Object Detection
- **Dataset**: MS COCO 2017
- **Task**: Object detection with focus on small object performance
- **Baselines**: DETR, YOLOv8
- **Multi-scale Setup**: Image pyramids at 3 resolution levels
- **Metrics**: mAP@0.5, mAP@0.5:0.95, mAP for small/medium/large objects
- **Rationale**: Object detection naturally benefits from multi-scale processing, especially for small objects

### Phase 2: Ablation & Robustness Studies

#### 2.1 Positional Encoding Comparison
- **Variants**: RoSE (rose_initial_scaling="log"), RoPE (rose_initial_scaling="rope"), Absolute PE, No PE
- **Configuration**: Simplified with new `rose_initial_scaling` parameter - RoPE mode is now just `rose_initial_scaling="rope"`
- **Evaluation**: Performance on both medical segmentation and object detection
- **Rationale**: Validate the effectiveness of spatially-aware rotary embeddings vs standard RoPE

#### 2.2 Attention Mechanism Analysis
- **Variants**: Dense SGA only, Sparse SGA only, Mixed (Dense+Sparse)
- **Analysis**: Attention pattern visualization, computational efficiency
- **Rationale**: Understand the contribution of spatial grouping attention mechanisms

#### 2.3 Resolution Strategy Comparison
- **Training Strategies**:
  - Single-resolution (256x256)
  - Dual-resolution (256x256 + 128x128)
  - Multi-resolution (256x256 + 128x128 + 64x64)
- **Testing**: Evaluate each model on all resolution combinations
- **Metric**: Performance degradation across scales
- **Rationale**: Evaluate model's ability to generalize across different input scales

## Baseline Performance References

To avoid unnecessary retraining, we use established baseline scores from literature:

### ISIC 2018 Segmentation
- **U-Net**: Dice ~0.847, IoU ~0.765 (Codella et al., 2019)
- **Swin-UNet**: Dice ~0.863, IoU ~0.781 (Cao et al., 2022)
- **TransUNet**: Dice ~0.855, IoU ~0.773 (Chen et al., 2021)

### MS COCO 2017 Detection
- **DETR**: mAP 42.0, mAP_small 20.5 (Carion et al., 2020)
- **YOLOv8**: mAP 50.2, mAP_small 31.8 (Ultralytics, 2023)

## Success Criteria

### Minimum Viable Results (Proof-of-Concept)
- **Medical Segmentation**: ≥2% improvement in Dice score over best baseline
- **Object Detection**: ≥5% improvement in small object mAP over best baseline
- **Ablations**: Clear evidence that each component (RoSE, SGA, multi-res) contributes positively
- **Robustness**: ≤10% performance degradation when testing on different resolutions

### Stretch Goals
- **Medical Segmentation**: ≥5% improvement in Dice score
- **Object Detection**: ≥10% improvement in small object mAP
- **Efficiency**: Competitive or better computational efficiency vs. baselines
- **Generalization**: Consistent improvements across multiple datasets

## Computational Resources

### Estimated Requirements
- **GPU Memory**: 8-16GB per experiment
- **Training Time**: 4-8 hours per model on modern GPUs
- **Total Experiments**: ~20-30 individual runs
- **Storage**: ~10GB for datasets, ~5GB for model checkpoints

### Recommended Hardware
- **Local Development**: Single RTX 3080/4080 (12-16GB VRAM)
- **Full Experiments**: Multi-GPU setup (4x RTX A6000 or similar)
- **Cluster**: SLURM/LSF with GPU nodes

## Expected Outcomes

Based on architectural design principles and preliminary testing:

1. **Multi-scale processing** should provide 2-5% improvements on tasks with objects at multiple scales
2. **RoSE embeddings** should improve spatial understanding, particularly for dense prediction tasks
3. **SGA mechanisms** should reduce computational cost while maintaining or improving accuracy
4. **Resolution robustness** should be significantly better than single-scale baselines

## Datasets and Preprocessing

### ISIC 2018 Skin Lesion Segmentation
- **Source**: https://challenge.isic-archive.com/data
- **Size**: 2,594 training images, 100 validation images
- **Download**: **Automatic** - Dataset is automatically downloaded to local storage
- **Fallback**: Kaggle API (`kaggle datasets download -d kmader/skin-cancer-mnist-ham10000`)
- **Preprocessing**: Resize to target resolution, normalize to [0,1], data augmentation
- **Splits**: 80% train, 20% validation
- **Local Storage**: `/tmp/datasets/isic2018/` (for fast training access)

### MS COCO 2017 Object Detection
- **Source**: https://cocodataset.org/#download
- **Size**: 118,287 training images, 5,000 validation images
- **Download**: **Automatic** - Dataset is automatically downloaded to local storage
- **Preprocessing**: Multi-scale resize, normalize, standard COCO augmentations
- **Focus**: Small object detection performance (area < 32²)
- **Local Storage**: `/tmp/datasets/coco2017/` (for fast training access)

### Storage Architecture
- **Local Data**: Datasets downloaded to `/tmp/datasets/` for optimal I/O performance
- **Network Results**: Checkpoints, logs, and results saved to `./results/` (persistent with repo)
- **Smart Caching**: Automatically checks for existing datasets to avoid re-downloading

## Citations

```bibtex
@article{codella2019skin,
  title={Skin lesion analysis toward melanoma detection: A challenge at the 2017 international symposium on biomedical imaging (isbi), hosted by the international skin imaging collaboration (isic)},
  author={Codella, Noel C and Gutman, David and Celebi, M Emre and others},
  journal={IEEE transactions on medical imaging},
  year={2019}
}

@article{cao2022swin,
  title={Swin-unet: Unet-like pure transformer for medical image segmentation},
  author={Cao, Hu and Wang, Yueyue and Chen, Joy and others},
  journal={Computer Vision and Image Understanding},
  year={2022}
}

@article{chen2021transunet,
  title={TransUNet: Transformers make strong encoders for medical image segmentation},
  author={Chen, Jieneng and Lu, Yongyi and Yu, Qihang and others},
  journal={arXiv preprint arXiv:2102.04306},
  year={2021}
}

@inproceedings{carion2020end,
  title={End-to-end object detection with transformers},
  author={Carion, Nicolas and Massa, Francisco and Synnaeve, Gabriel and others},
  booktitle={European conference on computer vision},
  year={2020}
}
```
