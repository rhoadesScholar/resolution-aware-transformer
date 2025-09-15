# Robustness and Scale Invariance Testing

This experiment evaluates the robustness of the Resolution Aware Transformer across different scales, resolutions, and perturbations to validate the model's scale invariance properties.

## Experiment Categories

### 1. Resolution Transfer Testing
- Train on one resolution, test on multiple scales
- Evaluate degradation patterns and adaptation capabilities
- Test both upscaling and downscaling scenarios

### 2. Scale Invariance Analysis
- Systematic evaluation across resolution ranges
- Object detection at varying scales within images
- Cross-scale consistency metrics

### 3. Robustness to Perturbations
- Noise robustness (Gaussian, impulse, etc.)
- Compression artifacts (JPEG quality levels)
- Geometric transformations (rotation, perspective)

## Key Metrics

- **Scale Consistency**: Performance variation across resolutions
- **Transfer Capability**: Zero-shot performance on unseen resolutions
- **Robustness Score**: Aggregate performance under perturbations
- **Efficiency**: Computational cost vs accuracy trade-offs

## Test Protocols

### Resolution Ladder Test
```python
# Test resolutions from 128x128 to 1024x1024
resolutions = [128, 192, 256, 384, 512, 768, 1024]
for res in resolutions:
    evaluate_model(model, dataset, image_size=res)
```

### Cross-Scale Evaluation
```python
# Train on 256x256, test on multiple scales
train_res = 256
test_resolutions = [128, 192, 256, 384, 512]
model = train_model(train_resolution=train_res)
for test_res in test_resolutions:
    results = evaluate_model(model, test_resolution=test_res)
```

## Expected Outcomes

1. **Resolution Transfer**: <10% performance drop within 2x scale range
2. **Scale Invariance**: Consistent object detection across scales
3. **Noise Robustness**: Graceful degradation under common corruptions
4. **Efficiency**: Linear scaling of compute with resolution

## Usage

```bash
cd experiments/robustness

# Run resolution transfer test
python resolution_transfer.py --model_path ../medical_segmentation/checkpoints/best_model.pth

# Run scale invariance analysis
python scale_invariance.py --model_path ../object_detection/checkpoints/best_model.pth

# Run robustness evaluation
python robustness_test.py --model_path models/pretrained.pth --corruption_types all
```
