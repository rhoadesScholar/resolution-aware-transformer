# Resolution Aware Transformer: Experiments

This directory contains the experimental validation of the Resolution Aware Transformer (RAT) architecture. Our experiments are designed to demonstrate proof-of-concept results showing the effectiveness of multi-scale processing, Rotary Spatial Embeddings (RoSE), and Spatial Grouping Attention (SGA) mechanisms.

## Experimental Overview

### Core Hypothesis
Multi-resolution processing with spatial awareness significantly improves performance on tasks requiring both global context and fine-grained detail, particularly in medical imaging and object detection scenarios.

### Key Research Questions
1. **Multi-scale Advantage**: Does multi-resolution processing outperform single-resolution approaches?
2. **Component Effectiveness**: How do RoSE and SGA components individually contribute to performance?
3. **Architectural Choices**: What are the optimal architectural parameters (blocks, attention types, feature dimensions)?
4. **Scale Robustness**: How robust is the model to resolution variations at test time?

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

### Phase 2: Ablation Studies

#### 2.1 Positional Encoding Comparison
- **Variants**: RoSE (learnable), RoPE (conventional), Absolute PE, No PE
- **Evaluation**: Performance on both medical segmentation and object detection
- **Rationale**: Validate the effectiveness of spatially-aware rotary embeddings

#### 2.2 Attention Mechanism Analysis
- **Variants**: Dense SGA only, Sparse SGA only, Mixed (Dense+Sparse), Standard Multi-Head Attention
- **Analysis**: Attention pattern visualization, computational efficiency
- **Rationale**: Understand the contribution of spatial grouping attention mechanisms

#### 2.3 Resolution Strategy Comparison
- **Training Strategies**:
  - Single-resolution (256x256)
  - Dual-resolution (256x256 + 128x128)
  - Multi-resolution (256x256 + 128x128 + 64x64)
- **Testing**: Evaluate each model on all resolution combinations
- **Rationale**: Core validation of multi-resolution hypothesis

#### 2.4 Architecture Depth Analysis
- **Variants**: 2, 4, 6 transformer blocks
- **Feature Dimensions**: 128, 256
- **Analysis**: Performance vs. computational cost trade-offs
- **Rationale**: Determine optimal architecture size for different tasks

### Phase 3: Robustness Analysis

#### 3.1 Resolution Transfer
- **Setup**: Train on one resolution, test on multiple resolutions
- **Resolutions**: 128, 256, 512, 1024 pixels
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
- **Training Time**:
  - Medical segmentation: 4-8 hours per configuration
  - Object detection: 12-24 hours per configuration
- **Total Compute**: ~200-300 GPU hours for complete experimental suite

### Resource Optimization
- **Dataset Subsets**: Initial experiments on 10-20% of data for rapid iteration
- **Early Stopping**: Stop training when trends are established
- **Pretrained Initialization**: Use ImageNet pretrained features where possible
- **Batch Size Optimization**: Maximize GPU utilization while maintaining reproducibility

## Implementation Strategy

### Code Organization
```
experiments/
├── medical_segmentation/     # ISIC 2018 experiments
├── object_detection/        # MS COCO experiments
├── ablations/              # Component analysis
├── robustness/             # Resolution transfer tests
├── common/                 # Shared utilities
│   ├── datasets.py         # Data loading utilities
│   ├── metrics.py          # Evaluation metrics
│   ├── models.py           # Model implementations
│   └── utils.py            # General utilities
└── results/                # Experimental outputs
```

### Reproducibility
- **Random Seeds**: Fixed seeds for all experiments
- **Environment**: Docker containers with pinned dependencies
- **Logging**: Comprehensive experiment tracking with Weights & Biases
- **Checkpoints**: Save model states for result reproduction

## Timeline

| Week | Phase | Deliverables |
|------|-------|-------------|
| 1-2 | Medical Segmentation | RAT vs. baselines on ISIC 2018 |
| 2-3 | Object Detection | RAT vs. baselines on MS COCO small objects |
| 3-4 | Core Ablations | RoSE, SGA, multi-res component analysis |
| 4-5 | Architecture Analysis | Optimal depth/width determination |
| 5-6 | Robustness + Writing | Resolution transfer + manuscript preparation |

## Expected Outcomes

### Technical Contributions
1. **Empirical validation** of multi-resolution transformer architecture
2. **Component analysis** showing individual contribution of RoSE and SGA
3. **Architecture guidelines** for optimal configuration
4. **Robustness characterization** across input scales

### Publication Strategy
- **Main Paper**: Focus on medical segmentation + core ablations
- **Supplementary**: Complete experimental results + additional baselines
- **Code Release**: Full experimental suite with pretrained models
- **Follow-up**: Object detection results as separate computer vision venue submission

## References

- Cao, H., et al. (2022). Swin-UNet: Unet-like Pure Transformer for Medical Image Segmentation. ECCV Workshop.
- Carion, N., et al. (2020). End-to-End Object Detection with Transformers. ECCV.
- Chen, J., et al. (2021). TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation. arXiv preprint.
- Codella, N., et al. (2019). Skin Lesion Analysis Toward Melanoma Detection 2018. arXiv preprint.
- Ultralytics (2023). YOLOv8: A New State-of-the-Art Computer Vision Model.
