#!/usr/bin/env python3
"""
Data Setup Script for Resolution Aware Transformer Experiments

This script downloads and prepares all datasets needed for the experiments:
- ISIC 2018 Skin Lesion Segmentation Challenge
- MS COCO 2017 Object Detection

Usage:
    python setup_datasets.py --data_dir /path/to/data --datasets all
    python setup_datasets.py --data_dir /path/to/data --datasets isic coco
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import List
import urllib.request
from urllib.parse import urljoin

import requests
from tqdm import tqdm


class DatasetDownloader:
    """Base class for dataset downloaders."""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_file(
        self, url: str, filename: str, description: str | None = None
    ) -> Path:
        """Download a file with progress bar."""
        filepath = self.data_dir / filename

        if filepath.exists():
            print(f"File {filename} already exists. Skipping download.")
            return filepath

        print(f"Downloading {description or filename}...")

        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        with (
            open(filepath, "wb") as f,
            tqdm(
                desc=filename,
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar,
        ):
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

        return filepath

    def extract_zip(self, zip_path: Path, extract_to: Path | None = None) -> Path:
        """Extract a zip file."""
        if extract_to is None:
            extract_to = zip_path.parent

        print(f"Extracting {zip_path.name}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)

        return extract_to


class ISICDownloader(DatasetDownloader):
    """Download and setup ISIC 2018 dataset."""

    def __init__(self, data_dir: str):
        super().__init__(data_dir)
        self.isic_dir = self.data_dir / "ISIC2018"
        self.isic_dir.mkdir(parents=True, exist_ok=True)

    def download(self):
        """Download ISIC 2018 dataset."""
        print("Setting up ISIC 2018 Skin Lesion Segmentation dataset...")

        # ISIC 2018 URLs (these are the official challenge URLs)
        urls = {
            "train_images": "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1-2_Training_Input.zip",
            "train_masks": "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1_Training_GroundTruth.zip",
            "val_images": "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1-2_Validation_Input.zip",
            "val_masks": "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1_Validation_GroundTruth.zip",
            "test_images": "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1-2_Test_Input.zip",
        }

        # Download all files
        downloaded_files = {}
        for name, url in urls.items():
            try:
                filepath = self.download_file(url, f"{name}.zip", f"ISIC 2018 {name}")
                downloaded_files[name] = filepath
            except Exception as e:
                print(f"Warning: Could not download {name}: {e}")
                print(f"You may need to manually download from: {url}")

        # Extract files and organize
        self._organize_isic_data(downloaded_files)
        self._create_data_splits()

        print(f"ISIC 2018 dataset setup complete in: {self.isic_dir}")

    def _organize_isic_data(self, downloaded_files: dict):
        """Organize ISIC data into expected directory structure."""
        # Create target directories
        images_dir = self.isic_dir / "images"
        masks_dir = self.isic_dir / "masks"
        images_dir.mkdir(exist_ok=True)
        masks_dir.mkdir(exist_ok=True)

        # Extract and move files
        for name, filepath in downloaded_files.items():
            if not filepath.exists():
                continue

            print(f"Extracting {filepath.name}...")
            extract_dir = self.isic_dir / f"temp_{name}"
            self.extract_zip(filepath, extract_dir)

            # Move files to appropriate directories
            if "images" in name:
                for img_file in extract_dir.rglob("*.jpg"):
                    shutil.move(str(img_file), str(images_dir / img_file.name))
            elif "masks" in name:
                for mask_file in extract_dir.rglob("*.png"):
                    # Rename mask files to match expected format
                    new_name = (
                        mask_file.stem.replace("_segmentation", "")
                        + "_segmentation.png"
                    )
                    shutil.move(str(mask_file), str(masks_dir / new_name))

            # Clean up temporary directory
            shutil.rmtree(extract_dir, ignore_errors=True)

    def _create_data_splits(self):
        """Create train/val/test splits."""
        from sklearn.model_selection import train_test_split

        images_dir = self.isic_dir / "images"
        masks_dir = self.isic_dir / "masks"

        # Get all image files that have corresponding masks
        image_files = []
        for img_file in images_dir.glob("*.jpg"):
            mask_file = masks_dir / f"{img_file.stem}_segmentation.png"
            if mask_file.exists():
                image_files.append(img_file.stem)

        # Create splits (70% train, 15% val, 15% test)
        indices = list(range(len(image_files)))
        train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

        splits = {"train": train_idx, "val": val_idx, "test": test_idx}

        # Save splits
        splits_file = self.isic_dir / "splits.json"
        with open(splits_file, "w") as f:
            json.dump(splits, f, indent=2)

        print(
            f"Created data splits: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}"
        )


class COCODownloader(DatasetDownloader):
    """Download and setup MS COCO 2017 dataset."""

    def __init__(self, data_dir: str):
        super().__init__(data_dir)
        self.coco_dir = self.data_dir / "COCO2017"
        self.coco_dir.mkdir(parents=True, exist_ok=True)

    def download(self):
        """Download COCO 2017 dataset."""
        print("Setting up MS COCO 2017 dataset...")

        # COCO 2017 URLs
        urls = {
            "train_images": "http://images.cocodataset.org/zips/train2017.zip",
            "val_images": "http://images.cocodataset.org/zips/val2017.zip",
            "test_images": "http://images.cocodataset.org/zips/test2017.zip",
            "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        }

        # Download files
        downloaded_files = {}
        for name, url in urls.items():
            try:
                filepath = self.download_file(url, f"{name}.zip", f"COCO 2017 {name}")
                downloaded_files[name] = filepath
            except Exception as e:
                print(f"Warning: Could not download {name}: {e}")
                print(f"You may need to manually download from: {url}")

        # Extract and organize
        self._organize_coco_data(downloaded_files)

        print(f"COCO 2017 dataset setup complete in: {self.coco_dir}")

    def _organize_coco_data(self, downloaded_files: dict):
        """Organize COCO data into expected directory structure."""
        # Extract all files
        for name, filepath in downloaded_files.items():
            if not filepath.exists():
                continue

            print(f"Extracting {filepath.name}...")
            self.extract_zip(filepath, self.coco_dir)

        # The COCO dataset should now be organized as:
        # COCO2017/
        #   ├── train2017/
        #   ├── val2017/
        #   ├── test2017/
        #   └── annotations/

        print("COCO dataset extracted and organized.")


def setup_sample_data(data_dir: str):
    """Create small sample datasets for quick testing."""
    print("Setting up sample datasets for quick testing...")

    data_path = Path(data_dir)

    # Create sample ISIC data
    sample_isic_dir = data_path / "sample_ISIC2018"
    sample_isic_dir.mkdir(parents=True, exist_ok=True)

    images_dir = sample_isic_dir / "images"
    masks_dir = sample_isic_dir / "masks"
    images_dir.mkdir(exist_ok=True)
    masks_dir.mkdir(exist_ok=True)

    # Create dummy images and masks for testing
    import torch
    from PIL import Image
    import numpy as np

    for i in range(10):
        # Create dummy RGB image
        dummy_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        img = Image.fromarray(dummy_img)
        img.save(images_dir / f"ISIC_sample_{i:04d}.jpg")

        # Create dummy binary mask
        dummy_mask = np.random.randint(0, 2, (256, 256), dtype=np.uint8) * 255
        mask = Image.fromarray(dummy_mask, mode="L")
        mask.save(masks_dir / f"ISIC_sample_{i:04d}_segmentation.png")

    # Create splits for sample data
    splits = {"train": list(range(7)), "val": [7, 8], "test": [9]}

    with open(sample_isic_dir / "splits.json", "w") as f:
        json.dump(splits, f, indent=2)

    print(f"Sample ISIC dataset created in: {sample_isic_dir}")


def main():
    parser = argparse.ArgumentParser(description="Setup datasets for RAT experiments")
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Base directory to store datasets"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["isic", "coco", "sample", "all"],
        default=["all"],
        help="Datasets to download and setup",
    )
    parser.add_argument(
        "--sample_only", action="store_true", help="Only create sample data for testing"
    )

    args = parser.parse_args()

    if args.sample_only or "sample" in args.datasets:
        setup_sample_data(args.data_dir)
        return

    datasets_to_setup = args.datasets
    if "all" in datasets_to_setup:
        datasets_to_setup = ["isic", "coco"]

    try:
        if "isic" in datasets_to_setup:
            downloader = ISICDownloader(args.data_dir)
            downloader.download()

        if "coco" in datasets_to_setup:
            downloader = COCODownloader(args.data_dir)
            downloader.download()

        print("\n" + "=" * 50)
        print("Dataset setup complete!")
        print(f"Data directory: {args.data_dir}")
        print("You can now run experiments with:")
        print(f"  python experiments/run_experiments.py --data_dir {args.data_dir}")
        print("=" * 50)

    except Exception as e:
        print(f"Error setting up datasets: {e}")
        print("\nIf automatic download fails, you may need to:")
        print("1. Manually download datasets from official sources")
        print("2. Check your internet connection")
        print("3. Verify you have sufficient disk space")
        sys.exit(1)


if __name__ == "__main__":
    main()
