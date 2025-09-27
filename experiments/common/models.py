"""Model implementations and wrappers for experiments."""

from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add the source directory to path to import our model
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

import logging

logger = logging.getLogger(__name__)

try:
    from resolution_aware_transformer import ResolutionAwareTransformer
except ImportError:
    logger.warning(
        "Could not import ResolutionAwareTransformer. " "Make sure it's installed."
    )
    ResolutionAwareTransformer = None


class SegmentationHead(nn.Module):
    """Segmentation head for converting transformer outputs to segmentation masks."""

    def __init__(
        self,
        feature_dims: int,
        num_classes: int = 1,
        upsampling_method: str = "bilinear",
    ):
        super().__init__()
        self.feature_dims = feature_dims
        self.num_classes = num_classes
        self.upsampling_method = upsampling_method

        # Simple decoder
        self.decoder = nn.Sequential(
            nn.Linear(feature_dims, feature_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dims // 2, feature_dims // 4),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dims // 4, num_classes),
        )

    def forward(
        self, features: torch.Tensor, target_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Convert transformer features to segmentation mask.

        Args:
            features: Transformer features [B, N, C]
            target_size: Target spatial size (H, W)

        Returns:
            Segmentation logits [B, num_classes, H, W]
        """
        B, N, C = features.shape

        # Decode features
        logits = self.decoder(features)  # [B, N, num_classes]

        # Reshape to spatial dimensions
        # Assume square spatial arrangement
        spatial_size = int(N**0.5)
        if spatial_size * spatial_size != N:
            # Handle non-square arrangements
            # This is a simplification - in practice, you'd need the
            # actual spatial dimensions
            spatial_size = int(N**0.5)
            if spatial_size * spatial_size < N:
                spatial_size += 1

        # Reshape and pad if necessary
        logits = logits.transpose(1, 2)  # [B, num_classes, N]

        try:
            logits = logits.view(B, self.num_classes, spatial_size, spatial_size)
        except RuntimeError:
            # Fallback: interpolate from current size
            logits = logits.view(B, self.num_classes, int(N**0.5), -1)

        # Upsample to target size
        if logits.shape[-2:] != target_size:
            logits = F.interpolate(
                logits,
                size=target_size,
                mode=self.upsampling_method,
                align_corners=False if self.upsampling_method == "bilinear" else None,
            )

        return logits


class DetectionHead(nn.Module):
    """Detection head for converting transformer outputs to object detection."""

    def __init__(
        self,
        feature_dims: int,
        num_classes: int = 80,  # COCO classes
        num_queries: int = 100,
    ):
        super().__init__()
        self.feature_dims = feature_dims
        self.num_classes = num_classes
        self.num_queries = num_queries

        # Query embeddings
        self.query_embed = nn.Embedding(num_queries, feature_dims)

        # Prediction heads
        self.class_head = nn.Linear(feature_dims, num_classes + 1)  # +1 for background
        self.bbox_head = nn.Linear(feature_dims, 4)  # x1, y1, x2, y2

        # Cross attention for queries
        self.cross_attn = nn.MultiheadAttention(
            feature_dims, num_heads=8, batch_first=True
        )

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Convert transformer features to detection outputs.

        Args:
            features: Transformer features [B, N, C]

        Returns:
            Dictionary with 'pred_logits' and 'pred_boxes'
        """
        B = features.shape[0]

        # Get query embeddings
        queries = self.query_embed.weight.unsqueeze(0).repeat(
            B, 1, 1
        )  # [B, num_queries, C]

        # Cross attention between queries and features
        attended_queries, _ = self.cross_attn(queries, features, features)

        # Predictions
        pred_logits = self.class_head(
            attended_queries
        )  # [B, num_queries, num_classes+1]
        pred_boxes = self.bbox_head(attended_queries).sigmoid()  # [B, num_queries, 4]

        return {"pred_logits": pred_logits, "pred_boxes": pred_boxes}


class RATSegmentationModel(nn.Module):
    """Resolution Aware Transformer for segmentation."""

    def __init__(
        self,
        spatial_dims: int = 2,
        input_features: int = 3,
        feature_dims: int = 128,
        num_blocks: int = 4,
        num_classes: int = 1,
        sga_attention_type: str = "dense",
        **kwargs,
    ):
        super().__init__()

        if ResolutionAwareTransformer is None:
            raise ImportError("ResolutionAwareTransformer not available")

        # Filter out parameters that are not supported by ResolutionAwareTransformer
        # and handle special parameter mappings
        filtered_kwargs = {}
        for key, value in kwargs.items():
            if key in ["multi_scale", "scales"]:
                # These are handled at the data loader level, not model level
                continue
            elif key == "sga_attention_type":
                # Map sga_attention_type to sga_attention_type for the transformer
                filtered_kwargs["sga_attention_type"] = value
            elif key == "positional_encoding":
                # Handle different positional encoding types
                if value == "rose":
                    filtered_kwargs["learnable_rose"] = True
                elif value == "rope":
                    # RoPE: Use standard rotary embeddings without spatial scaling
                    filtered_kwargs["learnable_rose"] = False
                    # Store this so we can handle it in forward pass
                    self.use_rope_mode = True
                elif value == "absolute":
                    # Absolute: Disable rotary embeddings entirely
                    filtered_kwargs["learnable_rose"] = False
                    filtered_kwargs["rotary_ratio"] = 0.0
                elif value == "none":
                    # No positional encoding
                    filtered_kwargs["learnable_rose"] = False
                    filtered_kwargs["rotary_ratio"] = 0.0
                else:
                    # Default to RoSE
                    filtered_kwargs["learnable_rose"] = True
            else:
                filtered_kwargs[key] = value

        # Set default for RoPE mode if not set
        self.use_rope_mode = getattr(self, "use_rope_mode", False)

        self.backbone = ResolutionAwareTransformer(
            spatial_dims=spatial_dims,
            input_features=input_features,
            feature_dims=feature_dims,
            num_blocks=num_blocks,
            sga_attention_type=sga_attention_type,
            **filtered_kwargs,
        )

        self.seg_head = SegmentationHead(feature_dims, num_classes)
        self.num_classes = num_classes

    def forward(
        self,
        x: Union[torch.Tensor, List[torch.Tensor]],
        target_size: Optional[Tuple[int, int]] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass for segmentation.

        Args:
            x: Input image(s)
            target_size: Target output size

        Returns:
            Segmentation logits
        """
        # Handle RoPE mode by providing uniform spacing to eliminate spatial scaling
        input_spacing = None
        if hasattr(self, "use_rope_mode") and self.use_rope_mode:
            if isinstance(x, list):
                # For multi-scale inputs, provide unit spacing for each scale
                input_spacing = [[1.0, 1.0] for _ in x]
            else:
                # For single input, provide unit spacing
                input_spacing = [1.0, 1.0]

        # Get transformer features
        outputs = self.backbone(x, input_spacing=input_spacing)

        if isinstance(x, list):
            # Multi-scale input
            results = []
            for i, output in enumerate(outputs):
                features = output["x_out"]  # [B, N, C]

                # Get target size from input if not provided
                if target_size is None:
                    input_size = x[i].shape[-2:]
                else:
                    input_size = target_size

                logits = self.seg_head(features, input_size)
                results.append(logits)

            return results
        else:
            # Single scale input
            features = outputs[0]["x_out"]  # [B, N, C]

            if target_size is None:
                target_size = x.shape[-2:]

            logits = self.seg_head(features, target_size)
            return logits


class RATDetectionModel(nn.Module):
    """Resolution Aware Transformer for object detection."""

    def __init__(
        self,
        spatial_dims: int = 2,
        input_features: int = 3,
        feature_dims: int = 256,
        num_blocks: int = 6,
        num_classes: int = 80,
        num_queries: int = 100,
        sga_attention_type: str = "dense",
        **kwargs,
    ):
        super().__init__()

        if ResolutionAwareTransformer is None:
            raise ImportError("ResolutionAwareTransformer not available")

        # Filter out parameters that are not supported by ResolutionAwareTransformer
        # and handle special parameter mappings
        filtered_kwargs = {}
        for key, value in kwargs.items():
            if key in [
                "multi_scale",
                "scales",
                "num_queries",
                "bbox_loss_coef",
                "class_loss_coef",
                "giou_loss_coef",
            ]:
                # These are handled at the model level or data loader level, not by the transformer
                continue
            elif key == "sga_attention_type":
                # Map sga_attention_type to sga_attention_type for the transformer
                filtered_kwargs["sga_attention_type"] = value
            elif key == "positional_encoding":
                # Handle different positional encoding types
                if value == "rose":
                    filtered_kwargs["learnable_rose"] = True
                elif value == "rope":
                    # RoPE: Use standard rotary embeddings without spatial scaling
                    filtered_kwargs["learnable_rose"] = False
                    # Store this so we can handle it in forward pass
                    self.use_rope_mode = True
                elif value == "absolute":
                    # Absolute: Disable rotary embeddings entirely
                    filtered_kwargs["learnable_rose"] = False
                    filtered_kwargs["rotary_ratio"] = 0.0
                elif value == "none":
                    # No positional encoding
                    filtered_kwargs["learnable_rose"] = False
                    filtered_kwargs["rotary_ratio"] = 0.0
                else:
                    # Default to RoSE
                    filtered_kwargs["learnable_rose"] = True
            else:
                filtered_kwargs[key] = value

        # Set default for RoPE mode if not set
        self.use_rope_mode = getattr(self, "use_rope_mode", False)

        self.backbone = ResolutionAwareTransformer(
            spatial_dims=spatial_dims,
            input_features=input_features,
            feature_dims=feature_dims,
            num_blocks=num_blocks,
            sga_attention_type=sga_attention_type,
            **filtered_kwargs,
        )

        self.detection_head = DetectionHead(feature_dims, num_classes, num_queries)
        self.num_classes = num_classes

    def forward(
        self, x: Union[torch.Tensor, List[torch.Tensor]]
    ) -> Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
        """
        Forward pass for detection.

        Args:
            x: Input image(s)

        Returns:
            Detection outputs
        """
        # Handle RoPE mode by providing uniform spacing to eliminate spatial scaling
        input_spacing = None
        if hasattr(self, "use_rope_mode") and self.use_rope_mode:
            if isinstance(x, list):
                # For multi-scale inputs, provide unit spacing for each scale
                input_spacing = [[1.0, 1.0] for _ in x]
            else:
                # For single input, provide unit spacing
                input_spacing = [1.0, 1.0]

        # Get transformer features
        outputs = self.backbone(x, input_spacing=input_spacing)

        if isinstance(x, list):
            # Multi-scale input - use features from largest scale
            features = outputs[0]["x_out"]  # [B, N, C]
        else:
            # Single scale input
            features = outputs[0]["x_out"]  # [B, N, C]

        # Get detection predictions
        predictions = self.detection_head(features)
        return predictions


# Baseline model implementations for comparison
class SimpleUNet(nn.Module):
    """U-Net implementation using torch.hub for baseline comparison."""

    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 1,
        init_features: int = 32,
        pretrained: bool = True,
    ):
        super().__init__()

        try:
            # Load UNet from torch hub (brain-segmentation-pytorch)
            self.unet = torch.hub.load(
                "mateuszbuda/brain-segmentation-pytorch",
                "unet",
                in_channels=input_channels,
                out_channels=num_classes,
                init_features=init_features,
                pretrained=pretrained,
                trust_repo=True,  # Required for loading from external repos
            )
            logger.info(
                f"Loaded UNet from torch.hub with {input_channels} input channels, {num_classes} output classes"
            )
        except Exception as e:
            logger.warning(f"Failed to load UNet from torch.hub: {e}")
            logger.info("Falling back to simple UNet implementation")
            # Fallback to a minimal UNet implementation
            self.unet = self._create_simple_unet(input_channels, num_classes)

    def _create_simple_unet(self, input_channels: int, num_classes: int) -> nn.Module:
        """Fallback simple UNet implementation."""
        return nn.Sequential(
            # Simple encoder-decoder structure
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Bottleneck
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            # Simple decoder (upsampling + conv)
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.unet(x)


def create_model(model_name: str, task: str, num_classes: int, **kwargs) -> nn.Module:
    """
    Create a model for experiments.

    Args:
        model_name: Name of the model ('rat', 'unet', etc.)
        task: Task type ('segmentation', 'detection')
        num_classes: Number of classes
        **kwargs: Additional model arguments

    Returns:
        Model instance
    """
    model_name = model_name.lower()
    task = task.lower()

    # Filter out parameters that are not for the model itself
    filtered_kwargs = kwargs.copy()
    for param in ["multi_scale", "scales", "name"]:
        filtered_kwargs.pop(param, None)

    if model_name == "rat":
        if task == "segmentation":
            return RATSegmentationModel(num_classes=num_classes, **filtered_kwargs)
        elif task == "detection":
            return RATDetectionModel(num_classes=num_classes, **filtered_kwargs)
        else:
            raise ValueError(f"Unknown task for RAT: {task}")

    elif model_name == "unet":
        if task == "segmentation":
            # Extract UNet-specific parameters
            input_channels = filtered_kwargs.get("input_features", 3)
            init_features = filtered_kwargs.get("init_features", 32)
            pretrained = filtered_kwargs.get("pretrained", True)
            return SimpleUNet(
                input_channels=input_channels,
                num_classes=num_classes,
                init_features=init_features,
                pretrained=pretrained,
            )
        else:
            raise ValueError(f"U-Net only supports segmentation, got: {task}")

    else:
        raise ValueError(f"Unknown model: {model_name}")


def create_rat_detection_model(**kwargs) -> nn.Module:
    """
    Create RAT model specifically for object detection.

    Args:
        **kwargs: Model configuration arguments

    Returns:
        RATDetectionModel instance
    """
    # Define parameters that should be passed to the RAT model
    # vs those that are for the detection head or training
    rat_params = [
        "spatial_dims",
        "input_features",
        "feature_dims",
        "num_blocks",
        "sga_attention_type",
        "positional_encoding",
        "num_heads",
        "learnable_rose",
        "mlp_ratio",
        "mlp_dropout",
        "proj_kernel_size",
        "proj_padding",
        "sga_kernel_size",
        "stride",
        "iters",
        "mlp_bias",
        "mlp_activation",
        "qkv_bias",
        "base_theta",
        "init_jitter_std",
        "rotary_ratio",
        "frequency_scaling",
        "leading_tokens",
        "spacing",
    ]

    # Filter parameters for the RAT model
    filtered_kwargs = {}
    for key, value in kwargs.items():
        if key in rat_params:
            filtered_kwargs[key] = value

    # Extract detection-specific parameters
    num_classes = kwargs.get("num_classes", 80)  # COCO default
    num_queries = kwargs.get("num_queries", 100)

    # Set defaults for detection model
    filtered_kwargs.setdefault("feature_dims", 256)
    filtered_kwargs.setdefault("num_blocks", 6)

    return RATDetectionModel(
        num_classes=num_classes, num_queries=num_queries, **filtered_kwargs
    )


def load_pretrained_model(
    model_path: str, model_name: str, task: str, num_classes: int, **kwargs
) -> nn.Module:
    """
    Load a pretrained model.

    Args:
        model_path: Path to model checkpoint
        model_name: Name of the model
        task: Task type
        num_classes: Number of classes
        **kwargs: Additional model arguments

    Returns:
        Loaded model
    """
    model = create_model(model_name, task, num_classes, **kwargs)

    if Path(model_path).exists():
        checkpoint = torch.load(model_path, map_location="cpu")

        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        logger.info(f"Loaded pretrained model from {model_path}")
    else:
        logger.info(f"Warning: Model checkpoint not found at {model_path}")

    return model


def save_model(
    model: nn.Module,
    save_path: str,
    epoch: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
    metrics: Optional[Dict[str, float]] = None,
):
    """
    Save model checkpoint.

    Args:
        model: Model to save
        save_path: Path to save checkpoint
        epoch: Current epoch
        optimizer: Optimizer state (optional)
        metrics: Training metrics (optional)
    """
    save_path_obj = Path(save_path)
    save_path_obj.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if metrics is not None:
        checkpoint["metrics"] = metrics

    torch.save(checkpoint, save_path_obj)
    logger.info(f"Saved model checkpoint to {save_path_obj}")


# Model configurations for different experiments
MODEL_CONFIGS = {
    "rat_small": {
        "feature_dims": 128,
        "num_blocks": 2,
        "num_heads": 8,
        "sga_attention_type": "dense",
    },
    "rat_base": {
        "feature_dims": 256,
        "num_blocks": 4,
        "num_heads": 16,
        "sga_attention_type": "dense",
    },
    "rat_large": {
        "feature_dims": 512,
        "num_blocks": 6,
        "num_heads": 32,
        "sga_attention_type": "dense",
    },
    "rat_sparse": {
        "feature_dims": 256,
        "num_blocks": 4,
        "num_heads": 16,
        "sga_attention_type": "sparse",
    },
    "unet_small": {
        "init_features": 16,
        "pretrained": True,
    },
    "unet_base": {
        "init_features": 32,
        "pretrained": True,
    },
    "unet_large": {
        "init_features": 64,
        "pretrained": False,  # Use larger model without pretrained weights
    },
}


def get_model_config(config_name: str) -> Dict:
    """Get predefined model configuration."""
    if config_name in MODEL_CONFIGS:
        return MODEL_CONFIGS[config_name].copy()
    else:
        raise ValueError(f"Unknown model config: {config_name}")
