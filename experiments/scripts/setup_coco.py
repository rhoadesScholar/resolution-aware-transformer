#!/usr/bin/env python3
"""
COCO 2017 Dataset Setup Script

This script downloads and organizes the MS COCO 2017 dataset for object detection experiments.

Official COCO website: https://cocodataset.org/

The script can either:
1. Download the full COCO 2017 dataset automatically
2. Organize manually downloaded files
3. Create a sample dataset for testing

Usage:
    # Automatic download (requires ~25GB disk space)
    python setup_coco.py --output_dir /path/to/data/COCO2017 --download

    # Organize manually downloaded files
    python setup_coco.py --downloads_dir /path/to/downloaded/files --output_dir /path/to/data/COCO2017

    # Create sample dataset for testing
    python setup_coco.py --output_dir /path/to/data/sample_COCO2017 --sample_only
"""

import argparse
import json
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Optional
import urllib.request

import requests
from tqdm import tqdm


class COCODatasetOrganizer:
    """Downloads and organizes COCO 2017 dataset."""

    def __init__(self, output_dir: str, downloads_dir: Optional[str] = None):
        self.output_dir = Path(output_dir)
        self.downloads_dir = Path(downloads_dir) if downloads_dir else None
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_dataset(self):
        """Download COCO 2017 dataset from official sources."""
        print("Downloading COCO 2017 dataset...")
        print(
            "Warning: This will download ~25GB of data. Ensure you have sufficient disk space and bandwidth."
        )

        # COCO 2017 download URLs
        urls = {
            "train2017.zip": "http://images.cocodataset.org/zips/train2017.zip",
            "val2017.zip": "http://images.cocodataset.org/zips/val2017.zip",
            "test2017.zip": "http://images.cocodataset.org/zips/test2017.zip",
            "annotations_trainval2017.zip": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        }

        # Create downloads directory
        downloads_dir = self.output_dir / "downloads"
        downloads_dir.mkdir(exist_ok=True)

        # Download each file
        downloaded_files = {}
        for filename, url in urls.items():
            filepath = downloads_dir / filename

            if filepath.exists():
                print(f"File {filename} already exists, skipping download.")
                downloaded_files[filename] = filepath
                continue

            try:
                print(f"Downloading {filename}...")
                self._download_file_with_progress(url, filepath)
                downloaded_files[filename] = filepath
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
                print(f"You can manually download from: {url}")

        # Organize the downloaded files
        self._organize_files(downloaded_files)

    def organize_existing_files(self):
        """Organize manually downloaded COCO files."""
        if not self.downloads_dir or not self.downloads_dir.exists():
            raise ValueError("Downloads directory not specified or doesn't exist")

        print(f"Organizing COCO files from: {self.downloads_dir}")

        # Find downloaded files
        expected_files = [
            "train2017.zip",
            "val2017.zip",
            "test2017.zip",
            "annotations_trainval2017.zip",
        ]

        downloaded_files = {}
        for filename in expected_files:
            filepath = self.downloads_dir / filename
            if filepath.exists():
                downloaded_files[filename] = filepath
                print(f"✓ Found: {filename}")
            else:
                print(f"✗ Missing: {filename}")

        if not downloaded_files:
            raise FileNotFoundError(
                f"No COCO zip files found in {self.downloads_dir}. "
                f"Please download files from https://cocodataset.org/#download"
            )

        self._organize_files(downloaded_files)

    def _download_file_with_progress(self, url: str, filepath: Path):
        """Download file with progress bar."""
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        with (
            open(filepath, "wb") as f,
            tqdm(
                desc=filepath.name,
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

    def _organize_files(self, downloaded_files: Dict[str, Path]):
        """Extract and organize COCO files."""
        print("Extracting and organizing COCO dataset...")

        for filename, filepath in downloaded_files.items():
            print(f"Extracting {filename}...")

            try:
                with zipfile.ZipFile(filepath, "r") as zip_ref:
                    # Extract to output directory
                    zip_ref.extractall(self.output_dir)

            except zipfile.BadZipFile:
                print(f"Error: {filename} appears to be corrupted. Please re-download.")
                continue

        # Verify the expected directory structure
        self._verify_structure()

        # Create dataset statistics
        self._create_dataset_info()

        print(f"COCO 2017 dataset organized in: {self.output_dir}")

    def _verify_structure(self):
        """Verify the COCO dataset directory structure."""
        expected_dirs = ["train2017", "val2017", "annotations"]
        optional_dirs = ["test2017"]

        print("Verifying dataset structure...")
        for dirname in expected_dirs:
            dirpath = self.output_dir / dirname
            if dirpath.exists():
                file_count = len(list(dirpath.glob("*")))
                print(f"✓ {dirname}: {file_count} files")
            else:
                print(f"✗ Missing required directory: {dirname}")

        for dirname in optional_dirs:
            dirpath = self.output_dir / dirname
            if dirpath.exists():
                file_count = len(list(dirpath.glob("*")))
                print(f"✓ {dirname}: {file_count} files")

    def _create_dataset_info(self):
        """Create dataset information file."""
        info = {"dataset": "COCO 2017", "splits": {}, "statistics": {}}

        # Count files in each split
        for split in ["train2017", "val2017", "test2017"]:
            split_dir = self.output_dir / split
            if split_dir.exists():
                image_count = len(list(split_dir.glob("*.jpg")))
                info["splits"][split] = {
                    "image_count": image_count,
                    "path": str(split_dir),
                }

        # Check annotations
        ann_dir = self.output_dir / "annotations"
        if ann_dir.exists():
            ann_files = list(ann_dir.glob("*.json"))
            info["annotations"] = {
                "files": [f.name for f in ann_files],
                "path": str(ann_dir),
            }

        # Save dataset info
        info_file = self.output_dir / "dataset_info.json"
        with open(info_file, "w") as f:
            json.dump(info, f, indent=2)

        print(f"Dataset info saved to: {info_file}")


def create_sample_coco_dataset(output_dir: str, num_samples_per_split: int = 10):
    """Create a small sample COCO dataset for testing."""
    print(
        f"Creating sample COCO dataset with {num_samples_per_split} samples per split..."
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create directory structure
    train_dir = output_path / "train2017"
    val_dir = output_path / "val2017"
    ann_dir = output_path / "annotations"

    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)
    ann_dir.mkdir(exist_ok=True)

    # Create sample images and annotations
    from PIL import Image, ImageDraw
    import numpy as np

    # Sample categories (simplified COCO categories)
    categories = [
        {"id": 1, "name": "person", "supercategory": "person"},
        {"id": 2, "name": "car", "supercategory": "vehicle"},
        {"id": 3, "name": "cat", "supercategory": "animal"},
    ]

    # Create annotations for each split
    for split, split_dir in [("train", train_dir), ("val", val_dir)]:
        images = []
        annotations = []
        ann_id = 1

        for i in range(num_samples_per_split):
            # Create synthetic image
            img_array = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)

            # Add some simple shapes as "objects"
            draw = ImageDraw.Draw(img)

            # Random rectangles and circles
            num_objects = np.random.randint(1, 4)
            img_annotations = []

            for _ in range(num_objects):
                # Random bounding box
                x1, y1 = np.random.randint(0, 400, 2)
                x2, y2 = x1 + np.random.randint(50, 200), y1 + np.random.randint(
                    50, 150
                )
                x2, y2 = min(x2, 640), min(y2, 480)

                # Draw object
                color = tuple(np.random.randint(0, 255, 3))
                if np.random.random() > 0.5:
                    draw.rectangle([x1, y1, x2, y2], fill=color)
                else:
                    draw.ellipse([x1, y1, x2, y2], fill=color)

                # Create annotation
                width, height = x2 - x1, y2 - y1
                area = width * height
                category_id = np.random.choice([cat["id"] for cat in categories])

                annotation = {
                    "id": ann_id,
                    "image_id": i + 1,
                    "category_id": category_id,
                    "bbox": [x1, y1, width, height],
                    "area": area,
                    "iscrowd": 0,
                }
                annotations.append(annotation)
                ann_id += 1

            # Save image
            img_filename = f"{i+1:012d}.jpg"
            img_path = split_dir / img_filename
            img.save(img_path)

            # Image info
            image_info = {
                "id": i + 1,
                "file_name": img_filename,
                "width": 640,
                "height": 480,
            }
            images.append(image_info)

        # Create COCO annotation format
        coco_format = {
            "info": {
                "description": "Sample COCO dataset for testing",
                "version": "1.0",
                "year": 2024,
            },
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }

        # Save annotations
        ann_filename = f"instances_{split}2017.json"
        ann_path = ann_dir / ann_filename
        with open(ann_path, "w") as f:
            json.dump(coco_format, f, indent=2)

    # Create dataset info
    info = {
        "dataset": "Sample COCO 2017",
        "splits": {
            "train2017": {"image_count": num_samples_per_split},
            "val2017": {"image_count": num_samples_per_split},
        },
        "note": "This is a synthetic sample dataset for testing purposes",
    }

    info_file = output_path / "dataset_info.json"
    with open(info_file, "w") as f:
        json.dump(info, f, indent=2)

    print(f"Sample COCO dataset created in: {output_path}")
    print(f"Train images: {num_samples_per_split}")
    print(f"Val images: {num_samples_per_split}")


def main():
    parser = argparse.ArgumentParser(description="Setup COCO 2017 dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for organized dataset",
    )
    parser.add_argument(
        "--downloads_dir",
        type=str,
        help="Directory containing manually downloaded COCO zip files",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download dataset automatically (requires ~25GB disk space)",
    )
    parser.add_argument(
        "--sample_only", action="store_true", help="Create only sample data for testing"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples per split for sample dataset",
    )

    args = parser.parse_args()

    if args.sample_only:
        create_sample_coco_dataset(args.output_dir, args.num_samples)
    elif args.download:
        organizer = COCODatasetOrganizer(args.output_dir)
        organizer.download_dataset()
    elif args.downloads_dir:
        organizer = COCODatasetOrganizer(args.output_dir, args.downloads_dir)
        organizer.organize_existing_files()
    else:
        print(
            "Error: Must specify either --download, --downloads_dir, or --sample_only"
        )
        print("\nOptions:")
        print("  --download: Automatically download COCO 2017 dataset (~25GB)")
        print("  --downloads_dir: Organize manually downloaded files")
        print("  --sample_only: Create small sample dataset for testing")
        return

    print("\nNext steps:")
    print(f"1. Update config files to point to: {args.output_dir}")
    print("2. Run experiments with: python experiments/run_experiments.py")


if __name__ == "__main__":
    main()
