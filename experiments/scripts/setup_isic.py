#!/usr/bin/env python3
"""
ISIC 2018 Dataset Setup Script

This script helps organize the ISIC 2018 Skin Lesion Segmentation dataset
after manual download from the official challenge website.

ISIC 2018 Challenge: https://challenge.isic-archive.com/task/1/

Manual Download Steps:
1. Go to https://challenge.isic-archive.com/task/1/
2. Register for the challenge
3. Download the following files:
   - ISIC2018_Task1-2_Training_Input.zip (Training Images)
   - ISIC2018_Task1_Training_GroundTruth.zip (Training Masks)
   - ISIC2018_Task1-2_Validation_Input.zip (Validation Images)
   - ISIC2018_Task1_Validation_GroundTruth.zip (Validation Masks)
   - ISIC2018_Task1-2_Test_Input.zip (Test Images) [optional]

Usage:
    # After downloading files to a directory:
    python setup_isic.py --downloads_dir /path/to/downloaded/files --output_dir /path/to/data/ISIC2018

    # Create only sample data for testing:
    python setup_isic.py --sample_only --output_dir /path/to/data/sample_ISIC2018
"""

import argparse
import json
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class ISICDatasetOrganizer:
    """Organizes ISIC 2018 dataset from downloaded files."""

    def __init__(self, downloads_dir: str, output_dir: str):
        self.downloads_dir = Path(downloads_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def organize_dataset(self):
        """Organize the ISIC dataset from downloaded zip files."""
        print("Organizing ISIC 2018 dataset...")

        # Expected file mappings
        file_mappings = {
            "ISIC2018_Task1-2_Training_Input.zip": "train_images",
            "ISIC2018_Task1_Training_GroundTruth.zip": "train_masks",
            "ISIC2018_Task1-2_Validation_Input.zip": "val_images",
            "ISIC2018_Task1_Validation_GroundTruth.zip": "val_masks",
            "ISIC2018_Task1-2_Test_Input.zip": "test_images",
        }

        # Check which files are available
        available_files = {}
        for filename, filetype in file_mappings.items():
            filepath = self.downloads_dir / filename
            if filepath.exists():
                available_files[filetype] = filepath
                print(f"✓ Found: {filename}")
            else:
                print(f"✗ Missing: {filename}")

        if not available_files:
            raise FileNotFoundError(
                f"No ISIC zip files found in {self.downloads_dir}. "
                f"Please download files manually from the ISIC challenge website."
            )

        # Create output directories
        images_dir = self.output_dir / "images"
        masks_dir = self.output_dir / "masks"
        images_dir.mkdir(exist_ok=True)
        masks_dir.mkdir(exist_ok=True)

        # Extract and organize files
        temp_dir = self.output_dir / "temp_extraction"
        temp_dir.mkdir(exist_ok=True)

        try:
            for filetype, filepath in available_files.items():
                print(f"Processing {filepath.name}...")
                self._extract_and_organize(
                    filepath, filetype, images_dir, masks_dir, temp_dir
                )

            # Create train/val/test splits
            self._create_splits(images_dir, masks_dir)

            print(f"Dataset organized successfully in: {self.output_dir}")
            self._print_dataset_stats(images_dir, masks_dir)

        finally:
            # Clean up temporary directory
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    def _extract_and_organize(
        self,
        zip_path: Path,
        filetype: str,
        images_dir: Path,
        masks_dir: Path,
        temp_dir: Path,
    ):
        """Extract zip file and organize contents."""
        # Extract to temporary directory
        extract_path = temp_dir / filetype
        extract_path.mkdir(exist_ok=True)

        print(f"  Extracting {zip_path.name}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)

        # Move files to appropriate directories
        if "images" in filetype:
            self._organize_images(extract_path, images_dir)
        elif "masks" in filetype:
            self._organize_masks(extract_path, masks_dir)

    def _organize_images(self, extract_path: Path, images_dir: Path):
        """Organize image files."""
        # Find all image files in extracted directory
        image_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            image_files.extend(extract_path.rglob(ext))

        print(f"  Moving {len(image_files)} images...")
        for img_file in tqdm(image_files, desc="Moving images"):
            dest_path = images_dir / img_file.name
            if not dest_path.exists():
                shutil.copy2(img_file, dest_path)

    def _organize_masks(self, extract_path: Path, masks_dir: Path):
        """Organize mask files."""
        # Find all mask files in extracted directory
        mask_files = []
        for ext in ["*.png", "*.jpg", "*.jpeg"]:
            mask_files.extend(extract_path.rglob(ext))

        print(f"  Moving {len(mask_files)} masks...")
        for mask_file in tqdm(mask_files, desc="Moving masks"):
            # Ensure mask filename follows expected convention
            if "_segmentation" not in mask_file.stem:
                new_name = f"{mask_file.stem}_segmentation{mask_file.suffix}"
            else:
                new_name = mask_file.name

            dest_path = masks_dir / new_name
            if not dest_path.exists():
                shutil.copy2(mask_file, dest_path)

    def _create_splits(self, images_dir: Path, masks_dir: Path):
        """Create train/val/test splits."""
        # Find all image-mask pairs
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        valid_pairs = []

        for img_file in image_files:
            # Look for corresponding mask
            mask_candidates = [
                masks_dir / f"{img_file.stem}_segmentation.png",
                masks_dir / f"{img_file.stem}_segmentation.jpg",
                masks_dir / f"{img_file.stem}.png",
                masks_dir / f"{img_file.stem}.jpg",
            ]

            for mask_path in mask_candidates:
                if mask_path.exists():
                    valid_pairs.append(img_file.stem)
                    break

        print(f"Found {len(valid_pairs)} valid image-mask pairs")

        if len(valid_pairs) == 0:
            print("Warning: No valid image-mask pairs found!")
            return

        # Create splits (70% train, 15% val, 15% test)
        indices = list(range(len(valid_pairs)))

        if len(indices) > 10:  # Only split if we have enough data
            train_idx, temp_idx = train_test_split(
                indices, test_size=0.3, random_state=42
            )
            val_idx, test_idx = train_test_split(
                temp_idx, test_size=0.5, random_state=42
            )
        else:
            # For small datasets, put most in train
            train_idx = indices[: max(1, int(0.7 * len(indices)))]
            val_idx = indices[
                len(train_idx) : len(train_idx) + max(1, int(0.15 * len(indices)))
            ]
            test_idx = indices[len(train_idx) + len(val_idx) :]

        splits = {
            "train": train_idx,
            "val": val_idx,
            "test": test_idx,
            "image_names": valid_pairs,  # Store the actual filenames for reference
        }

        # Save splits
        splits_file = self.output_dir / "splits.json"
        with open(splits_file, "w") as f:
            json.dump(splits, f, indent=2)

        print(
            f"Created splits: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}"
        )

    def _print_dataset_stats(self, images_dir: Path, masks_dir: Path):
        """Print dataset statistics."""
        num_images = len(list(images_dir.glob("*")))
        num_masks = len(list(masks_dir.glob("*")))

        print("\nDataset Statistics:")
        print(f"  Images: {num_images}")
        print(f"  Masks: {num_masks}")
        print(f"  Directory: {self.output_dir}")


def create_sample_dataset(output_dir: str, num_samples: int = 20):
    """Create a small sample dataset for testing."""
    print(f"Creating sample ISIC dataset with {num_samples} samples...")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    images_dir = output_path / "images"
    masks_dir = output_path / "masks"
    images_dir.mkdir(exist_ok=True)
    masks_dir.mkdir(exist_ok=True)

    # Create synthetic skin lesion-like images and masks
    for i in tqdm(range(num_samples), desc="Creating samples"):
        # Create a synthetic skin-like image
        img_array = np.random.randint(100, 200, (256, 256, 3), dtype=np.uint8)

        # Add some skin-like texture
        noise = np.random.normal(0, 10, (256, 256, 3))
        img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)

        # Create image
        img = Image.fromarray(img_array)
        img_path = images_dir / f"ISIC_sample_{i:07d}.jpg"
        img.save(img_path)

        # Create a synthetic lesion mask
        mask_array = np.zeros((256, 256), dtype=np.uint8)

        # Add random elliptical lesion
        center_x, center_y = np.random.randint(64, 192, 2)
        radius_x, radius_y = np.random.randint(20, 60, 2)

        y, x = np.ogrid[:256, :256]
        ellipse_mask = ((x - center_x) / radius_x) ** 2 + (
            (y - center_y) / radius_y
        ) ** 2 <= 1
        mask_array[ellipse_mask] = 255

        # Add some noise to mask boundary
        if np.random.random() > 0.3:  # 70% of samples have lesions
            mask = Image.fromarray(mask_array.astype(np.uint8))
        else:
            mask = Image.fromarray(np.zeros((256, 256), dtype=np.uint8))

        mask_path = masks_dir / f"ISIC_sample_{i:07d}_segmentation.png"
        mask.save(mask_path)

    # Create splits
    indices = list(range(num_samples))
    train_size = int(0.7 * num_samples)
    val_size = int(0.15 * num_samples)

    train_idx = indices[:train_size]
    val_idx = indices[train_size : train_size + val_size]
    test_idx = indices[train_size + val_size :]

    splits = {"train": train_idx, "val": val_idx, "test": test_idx}

    splits_file = output_path / "splits.json"
    with open(splits_file, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"Sample dataset created in: {output_path}")
    print(f"Splits: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")


def main():
    parser = argparse.ArgumentParser(description="Setup ISIC 2018 dataset")
    parser.add_argument(
        "--downloads_dir",
        type=str,
        help="Directory containing downloaded ISIC zip files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for organized dataset",
    )
    parser.add_argument(
        "--sample_only", action="store_true", help="Create only sample data for testing"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=20,
        help="Number of samples for sample dataset",
    )

    args = parser.parse_args()

    if args.sample_only:
        create_sample_dataset(args.output_dir, args.num_samples)
    else:
        if not args.downloads_dir:
            print("Error: --downloads_dir is required when not using --sample_only")
            print("\nTo manually download ISIC 2018 data:")
            print("1. Go to https://challenge.isic-archive.com/task/1/")
            print("2. Register and download the required zip files")
            print(
                "3. Run this script with --downloads_dir pointing to the download location"
            )
            return

        organizer = ISICDatasetOrganizer(args.downloads_dir, args.output_dir)
        organizer.organize_dataset()

    print("\nNext steps:")
    print(f"1. Update config files to point to: {args.output_dir}")
    print("2. Run experiments with: python run_experiments.py")


if __name__ == "__main__":
    main()
