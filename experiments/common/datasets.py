"""Dataset utilities for experiments."""

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class ISICDataset(Dataset):
    """ISIC 2018 dataset for skin lesion segmentation."""

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        image_size: Union[int, Tuple[int, int]] = 256,
        multi_scale: bool = False,
        scales: List[int] = [256, 128, 64],
        transform: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None,
    ):
        """
        Initialize ISIC dataset.

        Args:
            data_dir: Path to ISIC dataset directory
            split: 'train', 'val', or 'test'
            image_size: Target image size
            multi_scale: Whether to return multi-scale images
            scales: List of scales for multi-scale training
            transform: Image transforms
            mask_transform: Mask transforms
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = (
            image_size if isinstance(image_size, tuple) else (image_size, image_size)
        )
        self.multi_scale = multi_scale
        self.scales = scales
        self.transform = transform
        self.mask_transform = mask_transform

        # Load image and mask paths
        self._load_paths()

        # Default transforms
        if self.transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(self.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

        if self.mask_transform is None:
            self.mask_transform = transforms.Compose(
                [
                    transforms.Resize(
                        self.image_size,
                        interpolation=transforms.InterpolationMode.NEAREST,
                    ),
                    transforms.ToTensor(),
                ]
            )

    def _load_paths(self):
        """Load image and mask file paths."""
        # Expected directory structure:
        # data_dir/
        #   ├── images/
        #   ├── masks/
        #   └── splits.json (optional)

        images_dir = self.data_dir / "images"
        masks_dir = self.data_dir / "masks"

        if not images_dir.exists() or not masks_dir.exists():
            raise ValueError(f"Images or masks directory not found in {self.data_dir}")

        # Get all image files
        image_files = sorted(
            list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        )

        # Find corresponding mask files
        self.samples = []
        for img_path in image_files:
            mask_path = masks_dir / f"{img_path.stem}_segmentation.png"
            if not mask_path.exists():
                mask_path = masks_dir / f"{img_path.stem}.png"

            if mask_path.exists():
                self.samples.append((str(img_path), str(mask_path)))

        # Split data if splits file doesn't exist
        splits_file = self.data_dir / "splits.json"
        if splits_file.exists():
            with open(splits_file, "r") as f:
                splits = json.load(f)

            if self.split in splits:
                split_indices = splits[self.split]
                self.samples = [self.samples[i] for i in split_indices]
        else:
            # Create train/val/test splits (70/15/15)
            indices = list(range(len(self.samples)))
            train_idx, temp_idx = train_test_split(
                indices, test_size=0.3, random_state=42
            )
            val_idx, test_idx = train_test_split(
                temp_idx, test_size=0.5, random_state=42
            )

            split_dict = {"train": train_idx, "val": val_idx, "test": test_idx}

            if self.split in split_dict:
                split_indices = split_dict[self.split]
                self.samples = [self.samples[i] for i in split_indices]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        # Load image and mask
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.multi_scale:
            # Create multi-scale images
            images = []
            masks = []

            for scale in self.scales:
                scale_size = (scale, scale)

                # Scale transforms
                img_transform = transforms.Compose(
                    [
                        transforms.Resize(scale_size),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                )

                mask_transform = transforms.Compose(
                    [
                        transforms.Resize(
                            scale_size,
                            interpolation=transforms.InterpolationMode.NEAREST,
                        ),
                        transforms.ToTensor(),
                    ]
                )

                images.append(img_transform(image))
                masks.append(mask_transform(mask))

            return {
                "images": images,
                "masks": masks,
                "scales": self.scales,
                "image_path": img_path,
                "mask_path": mask_path,
            }
        else:
            # Single scale
            image = self.transform(image)
            mask = self.mask_transform(mask)

            return {
                "image": image,
                "mask": mask,
                "image_path": img_path,
                "mask_path": mask_path,
            }


class COCODataset(Dataset):
    """COCO dataset for object detection (simplified)."""

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        image_size: Union[int, Tuple[int, int]] = 512,
        multi_scale: bool = False,
        scales: List[int] = [512, 256, 128],
        transform: Optional[Callable] = None,
    ):
        """
        Initialize COCO dataset.

        Args:
            data_dir: Path to COCO dataset directory
            split: 'train', 'val', or 'test'
            image_size: Target image size
            multi_scale: Whether to return multi-scale images
            scales: List of scales for multi-scale training
            transform: Image transforms
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = (
            image_size if isinstance(image_size, tuple) else (image_size, image_size)
        )
        self.multi_scale = multi_scale
        self.scales = scales
        self.transform = transform

        # Load annotations
        self._load_annotations()

        # Default transform
        if self.transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(self.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

    def _load_annotations(self):
        """Load COCO annotations."""
        # This is a simplified implementation
        # In practice, use pycocotools for proper COCO loading

        ann_file = self.data_dir / f"annotations/instances_{self.split}2017.json"
        if not ann_file.exists():
            raise ValueError(f"Annotation file not found: {ann_file}")

        with open(ann_file, "r") as f:
            self.coco_data = json.load(f)

        # Create image index
        self.images = {img["id"]: img for img in self.coco_data["images"]}

        # Group annotations by image
        self.image_annotations = {}
        for ann in self.coco_data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.image_annotations:
                self.image_annotations[img_id] = []
            self.image_annotations[img_id].append(ann)

        # Filter images that have annotations
        self.image_ids = list(self.image_annotations.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]

        # Load image
        img_path = self.data_dir / f"{self.split}2017" / img_info["file_name"]
        image = Image.open(img_path).convert("RGB")

        # Get annotations
        annotations = self.image_annotations[img_id]

        # Extract bounding boxes and labels
        boxes = []
        labels = []
        areas = []

        for ann in annotations:
            bbox = ann["bbox"]  # [x, y, width, height]
            x1, y1, w, h = bbox
            x2, y2 = x1 + w, y1 + h

            boxes.append([x1, y1, x2, y2])
            labels.append(ann["category_id"])
            areas.append(ann["area"])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)
        areas = torch.tensor(areas, dtype=torch.float32)

        # Categorize by size (COCO definition)
        small_objects = areas < 32**2
        medium_objects = (areas >= 32**2) & (areas < 96**2)
        large_objects = areas >= 96**2

        if self.multi_scale:
            # Create multi-scale images
            images = []
            target_boxes = []

            orig_size = image.size  # (width, height)

            for scale in self.scales:
                scale_size = (scale, scale)

                # Scale image
                img_transform = transforms.Compose(
                    [
                        transforms.Resize(scale_size),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                )

                scaled_image = img_transform(image)
                images.append(scaled_image)

                # Scale bounding boxes
                scale_x = scale / orig_size[0]
                scale_y = scale / orig_size[1]

                scaled_boxes = boxes.clone()
                scaled_boxes[:, [0, 2]] *= scale_x
                scaled_boxes[:, [1, 3]] *= scale_y

                target_boxes.append(scaled_boxes)

            return {
                "images": images,
                "boxes": target_boxes,
                "labels": labels,
                "areas": areas,
                "small_objects": small_objects,
                "medium_objects": medium_objects,
                "large_objects": large_objects,
                "scales": self.scales,
                "image_id": img_id,
            }
        else:
            # Single scale
            image = self.transform(image)

            # Scale boxes to match resized image
            orig_size = img_info["width"], img_info["height"]
            scale_x = self.image_size[0] / orig_size[0]
            scale_y = self.image_size[1] / orig_size[1]

            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y

            return {
                "image": image,
                "boxes": boxes,
                "labels": labels,
                "areas": areas,
                "small_objects": small_objects,
                "medium_objects": medium_objects,
                "large_objects": large_objects,
                "image_id": img_id,
            }


def create_dataloaders(
    dataset_name: str,
    data_dir: str,
    batch_size: int = 16,
    num_workers: int = 4,
    multi_scale: bool = False,
    image_size: int = 256,
) -> Dict[str, DataLoader]:
    """
    Create dataloaders for different datasets.

    Args:
        dataset_name: Name of dataset ('isic' or 'coco')
        data_dir: Path to dataset directory
        batch_size: Batch size
        num_workers: Number of data loading workers
        multi_scale: Whether to use multi-scale training
        image_size: Image size

    Returns:
        Dictionary of dataloaders for train/val/test splits
    """
    dataloaders = {}

    if dataset_name.lower() == "isic":
        for split in ["train", "val", "test"]:
            dataset = ISICDataset(
                data_dir=data_dir,
                split=split,
                image_size=image_size,
                multi_scale=multi_scale,
            )

            # Use different batch sizes for train vs eval
            bs = batch_size if split == "train" else batch_size * 2
            shuffle = split == "train"

            dataloader = DataLoader(
                dataset,
                batch_size=bs,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=multi_scale_collate_fn if multi_scale else None,
            )

            dataloaders[split] = dataloader

    elif dataset_name.lower() == "coco":
        for split in ["train", "val"]:
            dataset = COCODataset(
                data_dir=data_dir,
                split=split,
                image_size=image_size,
                multi_scale=multi_scale,
            )

            bs = batch_size if split == "train" else batch_size * 2
            shuffle = split == "train"

            dataloader = DataLoader(
                dataset,
                batch_size=bs,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=detection_collate_fn,
            )

            dataloaders[split] = dataloader

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return dataloaders


def multi_scale_collate_fn(batch):
    """Collate function for multi-scale batches."""
    if "images" in batch[0]:  # Multi-scale
        # Group by scale
        num_scales = len(batch[0]["images"])
        images_by_scale = [[] for _ in range(num_scales)]
        masks_by_scale = [[] for _ in range(num_scales)]

        for sample in batch:
            for i, (img, mask) in enumerate(zip(sample["images"], sample["masks"])):
                images_by_scale[i].append(img)
                masks_by_scale[i].append(mask)

        # Stack tensors for each scale
        batched_images = [torch.stack(imgs) for imgs in images_by_scale]
        batched_masks = [torch.stack(masks) for masks in masks_by_scale]

        return {
            "images": batched_images,
            "masks": batched_masks,
            "scales": batch[0]["scales"],
        }
    else:
        # Single scale - use default collate
        return torch.utils.data.default_collate(batch)


def detection_collate_fn(batch):
    """Collate function for detection data."""
    # Detection batches need special handling due to variable number of objects

    if "images" in batch[0]:  # Multi-scale
        num_scales = len(batch[0]["images"])
        images_by_scale = [[] for _ in range(num_scales)]

        for sample in batch:
            for i, img in enumerate(sample["images"]):
                images_by_scale[i].append(img)

        batched_images = [torch.stack(imgs) for imgs in images_by_scale]

        # Keep boxes and labels as lists (variable length)
        boxes = [sample["boxes"] for sample in batch]
        labels = [sample["labels"] for sample in batch]

        return {
            "images": batched_images,
            "boxes": boxes,
            "labels": labels,
            "scales": batch[0]["scales"],
        }
    else:
        # Single scale
        images = torch.stack([sample["image"] for sample in batch])
        boxes = [sample["boxes"] for sample in batch]
        labels = [sample["labels"] for sample in batch]

        return {"images": images, "boxes": boxes, "labels": labels}


def download_dataset(dataset_name: str, download_dir: str) -> str:
    """
    Download dataset if not already present.

    Args:
        dataset_name: Name of dataset to download
        download_dir: Directory to download to

    Returns:
        Path to downloaded dataset
    """
    download_dir_path = Path(download_dir)
    download_dir_path.mkdir(parents=True, exist_ok=True)

    if dataset_name.lower() == "isic":
        # ISIC 2018 dataset
        dataset_dir = download_dir_path / "ISIC2018"

        if not dataset_dir.exists():
            print("ISIC 2018 dataset not found. Please download manually from:")
            print("https://challenge.isic-archive.com/data/")
            print("Required files:")
            print("- ISIC2018_Task1-2_Training_Input (images)")
            print("- ISIC2018_Task1_Training_GroundTruth (segmentation masks)")
            print("- ISIC2018_Task1-2_Validation_Input (validation images)")
            print("- ISIC2018_Task1_Validation_GroundTruth (validation masks)")

        return str(dataset_dir)

    elif dataset_name.lower() == "coco":
        # COCO 2017 dataset
        dataset_dir = download_dir_path / "COCO2017"

        if not dataset_dir.exists():
            print("COCO 2017 dataset not found. Please download manually from:")
            print("https://cocodataset.org/#download")
            print("Required files:")
            print("- 2017 Train images")
            print("- 2017 Val images")
            print("- 2017 Train/Val annotations")

        return str(dataset_dir)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


# Dataset statistics for normalization
DATASET_STATS = {
    "isic": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    "coco": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
}
