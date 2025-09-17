# Medical Segmentation Experiment

This experiment evaluates the Resolution Aware Transformer on the ISIC 2018 skin lesion segmentation task.

## Experiment Setup

### Dataset
- **ISIC 2018 Task 1**: Skin Lesion Analysis Towards Melanoma Detection
- **Task**: Binary segmentation of melanoma lesions
- **Images**: ~2,594 training images, ~100 validation images, ~1,000 test images
- **Format**: RGB dermoscopy images with binary segmentation masks

### Models Compared
1. **Resolution Aware Transformer (RAT)** - Our proposed method
   - Single-scale: 256x256 input
   - Multi-scale: 256x256 + 128x128 + 64x64 inputs
   - Variants: Dense vs. Sparse attention, different depths

2. **Baseline Models**:
   - U-Net (established baseline for medical segmentation)
   - Swin-UNet (transformer-based baseline)
   - TransUNet (hybrid CNN-transformer)

### Evaluation Metrics
- **Dice Coefficient**: Primary metric for segmentation overlap
- **IoU (Jaccard Index)**: Intersection over Union
- **Sensitivity**: True positive rate
- **Specificity**: True negative rate

### Multi-Scale Strategy
- **Single-resolution**: Train and test on 256x256 images only
- **Multi-resolution**: Train on image pyramids (256x256, 128x128, 64x64)
- **Mixed evaluation**: Test single-res models on multi-res inputs and vice versa

## Expected Results

Based on baseline performance from literature:
- U-Net: Dice ~0.847, IoU ~0.765
- Swin-UNet: Dice ~0.863, IoU ~0.781
- TransUNet: Dice ~0.855, IoU ~0.773

**Target improvements**:
- RAT (single-scale): Dice ≥0.87 (+2% over best baseline)
- RAT (multi-scale): Dice ≥0.88 (+3% over best baseline)

## Running the Experiment

```bash
cd experiments/medical_segmentation
python train.py --config configs/rat_multiscale.yaml
python evaluate.py --model_path checkpoints/rat_multiscale_best.pth
python ablation_study.py --config configs/ablation.yaml
```

## Configuration Files

- `configs/rat_single.yaml`: Single-scale RAT configuration
- `configs/rat_multiscale.yaml`: Multi-scale RAT configuration
- `configs/ablation.yaml`: Ablation study configurations
- `configs/baseline.yaml`: Baseline model configurations
