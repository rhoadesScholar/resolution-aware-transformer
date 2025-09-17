# Data Setup Scripts

This directory contains scripts to download, organize, and configure datasets for the Resolution Aware Transformer (RAT) experiments.

## Quick Start

### For Testing (Recommended First Step)
```bash
# Set up sample datasets for quick testing
python scripts/setup_experiments.py --quick --data_dir ./data
```

### For Full Experiments
```bash
# 1. Download ISIC 2018 manually from https://challenge.isic-archive.com/task/1/
# 2. Download COCO 2017 files or use automatic download

# Option A: With manual downloads
python scripts/setup_experiments.py \
    --isic_downloads /path/to/isic/downloads \
    --coco_downloads /path/to/coco/downloads \
    --data_dir ./data

# Option B: With automatic COCO download (~25GB)
python scripts/setup_experiments.py \
    --isic_downloads /path/to/isic/downloads \
    --download_coco \
    --data_dir ./data
```

## Individual Scripts

### 1. `setup_experiments.py` (Master Script)
The main orchestration script that handles the complete setup process.

**Features:**
- Environment setup and package installation
- Dataset organization
- Configuration file updates
- Verification tests
- User guidance

**Usage:**
```bash
# Quick setup with sample data
python scripts/setup_experiments.py --quick

# Full setup
python scripts/setup_experiments.py --isic_downloads /path --coco_downloads /path --data_dir ./data

# Help
python scripts/setup_experiments.py --help
```

### 2. `setup_isic.py` (ISIC Dataset)
Handles ISIC 2018 Skin Lesion Segmentation dataset setup.

**Features:**
- Organizes manually downloaded ISIC files
- Creates proper directory structure
- Generates train/val/test splits
- Creates sample datasets for testing

**Manual Download Steps:**
1. Go to https://challenge.isic-archive.com/task/1/
2. Register for the challenge
3. Download these files:
   - `ISIC2018_Task1-2_Training_Input.zip` (Training Images)
   - `ISIC2018_Task1_Training_GroundTruth.zip` (Training Masks)
   - `ISIC2018_Task1-2_Validation_Input.zip` (Validation Images)
   - `ISIC2018_Task1_Validation_GroundTruth.zip` (Validation Masks)
   - `ISIC2018_Task1-2_Test_Input.zip` (Test Images) [optional]

**Usage:**
```bash
# Organize manually downloaded files
python scripts/setup_isic.py \
    --downloads_dir /path/to/isic/downloads \
    --output_dir ./data/ISIC2018

# Create sample data for testing
python scripts/setup_isic.py \
    --sample_only \
    --output_dir ./data/sample_ISIC2018 \
    --num_samples 50
```

### 3. `setup_coco.py` (COCO Dataset)
Handles MS COCO 2017 object detection dataset setup.

**Features:**
- Automatic download from official sources
- Organization of manually downloaded files
- Verification of dataset structure
- Sample dataset creation

**Usage:**
```bash
# Automatic download (~25GB)
python scripts/setup_coco.py \
    --download \
    --output_dir ./data/COCO2017

# Organize manually downloaded files
python scripts/setup_coco.py \
    --downloads_dir /path/to/coco/downloads \
    --output_dir ./data/COCO2017

# Create sample data for testing
python scripts/setup_coco.py \
    --sample_only \
    --output_dir ./data/sample_COCO2017 \
    --num_samples 20
```

### 4. `update_configs.py` (Configuration Update)
Updates experiment configuration files with correct dataset paths.

**Features:**
- Updates all YAML config files
- Creates quick test configurations
- Verifies data paths
- Handles nested configuration updates

**Usage:**
```bash
# Update configs with base data directory
python scripts/update_configs.py --data_dir ./data

# Update with specific paths
python scripts/update_configs.py \
    --isic_dir ./data/ISIC2018 \
    --coco_dir ./data/COCO2017 \
    --create_quick_configs
```

## Expected Directory Structure

After running the setup scripts, your data directory will look like:

```
data/
├── ISIC2018/                     # Real ISIC dataset
│   ├── images/                   # Training images (.jpg)
│   ├── masks/                    # Segmentation masks (.png)
│   ├── splits.json              # Train/val/test split indices
│   └── dataset_info.json       # Dataset metadata
├── sample_ISIC2018/             # Sample ISIC for testing
│   ├── images/                  # Sample images
│   ├── masks/                   # Sample masks
│   └── splits.json
├── COCO2017/                    # Real COCO dataset
│   ├── train2017/               # Training images
│   ├── val2017/                 # Validation images
│   ├── test2017/                # Test images (optional)
│   ├── annotations/             # COCO annotations
│   └── dataset_info.json
└── sample_COCO2017/             # Sample COCO for testing
    ├── train2017/
    ├── val2017/
    ├── annotations/
    └── dataset_info.json
```

## Configuration Files Updated

The scripts automatically update these configuration files:

- `experiments/medical_segmentation/configs/*.yaml`
- `experiments/object_detection/configs/*.yaml`
- `experiments/ablations/configs/*.yaml`
- `experiments/configs/quick_test_*.yaml` (created for testing)

## Troubleshooting

### Common Issues

1. **Download Failures**
   - Check internet connection
   - Verify sufficient disk space (~30GB for full datasets)
   - For ISIC: manually download from the challenge website
   - For COCO: try the manual download option

2. **Permission Errors**
   - Ensure write permissions in the target directory
   - Use `sudo` if necessary (not recommended)

3. **Import Errors**
   - Make sure the package is installed: `pip install -e .`
   - Check Python environment and dependencies

4. **Configuration Issues**
   - Manually verify paths in config files
   - Re-run `update_configs.py` if needed

### Disk Space Requirements

- **Sample datasets**: ~100MB
- **ISIC 2018**: ~2GB
- **COCO 2017**: ~25GB
- **Total recommended**: 30GB free space

### Manual Dataset Download Links

- **ISIC 2018**: https://challenge.isic-archive.com/task/1/
- **COCO 2017**: https://cocodataset.org/#download

## Next Steps

After successful setup:

1. **Quick Test:**
   ```bash
   cd experiments
   python run_experiments.py --experiments medical_seg --quick
   ```

2. **Full Experiments:**
   ```bash
   python run_experiments.py --experiments all
   ```

3. **Check Logs:**
   Monitor `experiments/results/` for experiment outputs and logs.

## Support

If you encounter issues:

1. Check this README and script help messages
2. Verify your Python environment and dependencies
3. Ensure sufficient disk space and permissions
4. Check dataset download sources for any changes