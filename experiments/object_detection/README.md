# Object Detection with Resolution Aware Transformer

This experiment evaluates the RAT on object detection using MS COCO 2017. We test the model's ability to detect objects at multiple scales and compare against DETR and other transformer-based detectors.

## Experiment Setup

- **Dataset**: MS COCO 2017 (subset for efficiency)
- **Task**: Object detection and localization
- **Metric**: mAP@0.5, mAP@0.5:0.95, FPS
- **Baseline**: DETR (Detection Transformer)

## Configurations

### 1. Single Scale Detection (`single_scale.yaml`)
- Fixed 800x800 input resolution
- Standard DETR-style architecture with RAT backbone
- Baseline comparison

### 2. Multi-Scale Detection (`multi_scale.yaml`)
- Multi-resolution processing: 800x800, 400x400, 200x200
- Cross-scale feature fusion
- Enhanced small object detection

## Key Hypotheses

1. **Multi-scale processing improves small object detection**
2. **Spatial grouping attention reduces computational cost vs dense attention**
3. **RoSE provides better spatial understanding than standard positional encoding**

## Success Criteria

- mAP@0.5 improvement ≥ 2% over DETR baseline
- Competitive inference speed (≥ 15 FPS)
- Strong small object performance (mAP_s ≥ baseline + 3%)

## Usage

```bash
# Train single-scale model
cd experiments/object_detection
python train.py --config configs/single_scale.yaml --data_dir /path/to/coco

# Train multi-scale model
python train.py --config configs/multi_scale.yaml --data_dir /path/to/coco

# Evaluate model
python evaluate.py --model_path checkpoints/best_model.pth --config configs/multi_scale.yaml
```

## Expected Outcomes

Based on our architectural advantages:
- **Small objects**: +3-5% mAP improvement due to multi-scale processing
- **Medium/Large objects**: Competitive performance with efficiency gains
- **Speed**: 15-20 FPS vs DETR's 10-12 FPS due to sparse attention
- **Memory**: 20-30% reduction in peak memory usage
