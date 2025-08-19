"""Core functionality for resolution_aware_transformer."""

from typing import Optional, Sequence, Type

from spatial_grouping_attention import (
    DenseSpatialGroupingAttention,
    SparseSpatialGroupingAttention,
)
from spatial_grouping_attention.utils import to_list, to_tuple
import torch

# TODO: Import MultiResolutionDeformableAttention when available
# from some_module import MultiResolutionDeformableAttention


class ResolutionAwareTransformer(torch.nn.Module):
    """Resolution Aware Transformer class
    Args:
        spatial_dims: The spatial dimensions of the input images. (e.g. 2 or 3)
        feature_dims: The dimensionality of the embedding space.
        num_blocks: The number of spatial grouping attention and
                    multi-res deformable attention pairs
        attention_type: The type of attention to use (default is "dense").
                        Options are "dense" or "sparse"
        kernel_size: Size of the convolutional kernel for strided
                     convolutions producing the lower res embeddings in each
                     spatial grouping attention layer
        stride: Stride for the strided convolutions (default is half
                kernel size)
        num_heads: Number of attention heads
        iters: Number of iterations for the attention mechanism
               (default is 3)
        mlp_ratio: Ratio of hidden to input/output dimensions in the MLP
                   (default is 4)
        mlp_dropout: Dropout rate for the MLP (default is 0.0)
        mlp_bias: Whether to use bias in the MLP (default is True)
        mlp_activation: Activation function for the MLP (default is GELU)
        qkv_bias: Whether to use bias in the query/key/value linear
                  layers (default is True)
        base_theta: Base theta value for the rotary position embedding
                    (default is 1e4)
        learnable_rose: Whether to use learnable rotary spatial embeddings
                        (default is True)
        init_jitter_std: Standard deviation for initial jitter in the rotary
                         embeddings (default is 0.02)
        spacing: Default real-world pixel spacing for the input data
                 (default is None, which uses a default spacing of 1.0 for
                 all dimensions). Can be specified at initialization or passed
                 during the forward pass.
    """

    def __init__(
        self,
        spatial_dims: int = 2,
        input_features: int = 3,
        proj_kernel_size: int | Sequence[int] = 7,
        proj_padding: int | Sequence[int] = 3,
        feature_dims: int = 128,
        num_blocks: int = 4,
        attention_type: str = "dense",
        kernel_size: int | Sequence[int] = 7,
        stride: Optional[int | Sequence[int]] = None,
        num_heads: int = 16,
        iters: int = 3,
        mlp_ratio: float | int = 4,
        mlp_dropout: float = 0.0,
        mlp_bias: bool = True,
        mlp_activation: Type[torch.nn.Module] = torch.nn.GELU,
        qkv_bias: bool = True,
        base_theta: float = 1e4,
        learnable_rose: bool = True,
        init_jitter_std: float = 0.02,
        spacing: Optional[float | Sequence[float]] = None,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.input_features = input_features
        self.proj_kernel_size = to_tuple(
            proj_kernel_size, spatial_dims, dtype_caster=int, allow_nested=False
        )
        self.proj_padding = to_tuple(
            proj_padding, spatial_dims, dtype_caster=int, allow_nested=False
        )
        self.feature_dims = feature_dims
        self.num_blocks = num_blocks
        self.attention_type = to_tuple(attention_type, num_blocks, allow_nested=False)
        self.kernel_size = to_tuple(
            kernel_size, spatial_dims, dtype_caster=int, allow_nested=False
        )
        if stride is None:
            # Default stride is half kernel size, minimum 1
            self.stride = tuple(
                max(1, k // 2) for k in self.kernel_size  # type: ignore
            )
        else:
            self.stride = to_tuple(
                stride, spatial_dims, dtype_caster=int, allow_nested=False
            )
        self.num_heads = num_heads
        self.iters = iters
        self.mlp_ratio = mlp_ratio
        self.mlp_dropout = mlp_dropout
        self.mlp_bias = mlp_bias
        self.mlp_activation = mlp_activation
        self.qkv_bias = qkv_bias
        self.base_theta = base_theta
        self.learnable_rose = learnable_rose
        self.init_jitter_std = init_jitter_std
        if spacing is None:
            spacing = 1.0
        self._default_spacing = to_tuple(spacing, spatial_dims)

        # Build model
        conv = {1: torch.nn.Conv1d, 2: torch.nn.Conv2d, 3: torch.nn.Conv3d}[
            spatial_dims
        ]
        self.init_proj = conv(
            in_channels=input_features,
            out_channels=feature_dims,
            kernel_size=proj_kernel_size,
            padding=proj_padding,
        )
        sga_layers = []
        mrda_layers = []
        sga_kwargs = {
            "feature_dims": feature_dims,
            "spatial_dims": spatial_dims,
            "kernel_size": self.kernel_size,
            "num_heads": num_heads,
            "stride": self.stride,
            "iters": iters,
            "mlp_ratio": mlp_ratio,
            "mlp_dropout": mlp_dropout,
            "mlp_bias": mlp_bias,
            "mlp_activation": mlp_activation,
            "qkv_bias": qkv_bias,
            "base_theta": base_theta,
            "learnable_rose": learnable_rose,
            "init_jitter_std": init_jitter_std,
            "spacing": self._default_spacing,
        }
        # TODO: Define mrda_kwargs when MRDA is available
        for n in range(num_blocks):
            sga = (
                DenseSpatialGroupingAttention
                if self.attention_type[n] == "dense"
                else SparseSpatialGroupingAttention
            )
            sga_layers.append(sga(**sga_kwargs))
            # TODO: Uncomment when MRDA is available
            # mrda_layers.append(MultiResolutionDeformableAttention(...))

        self.sga_layers = torch.nn.ModuleList(sga_layers)
        self.mrda_layers = torch.nn.ModuleList(mrda_layers)

        # Initialize weights
        self.apply(init_transformer_weights)

    def _prepare_inputs(self, x, input_spacing, mask):
        if input_spacing is None:
            input_spacing = self._default_spacing
        input_spacing = to_list(input_spacing, self.spatial_dims)

        # Make everything lists if not already (for image res pyramid)
        if isinstance(x, torch.Tensor):
            x = [x]
        if isinstance(mask, torch.Tensor):
            mask = [mask]
        if not isinstance(input_spacing[0], Sequence):
            input_spacing = [input_spacing]

        # Get input grid shapes
        input_grid_shapes = [tuple(_x.shape[-self.spatial_dims :]) for _x in x]

        # Dilate masks if necessary to handle init_proj kernel size
        if mask is not None:
            for i, m in enumerate(mask):
                mask[i] = self.dilate_mask(m)

        return x, input_spacing, mask, input_grid_shapes

    def dilate_mask(self, mask: torch.Tensor) -> torch.Tensor:
        if all([k == 1 for k in self.proj_kernel_size]):
            return mask
        else:
            if not hasattr(self, "_dilator"):
                pooler = {
                    1: torch.nn.MaxPool1d,
                    2: torch.nn.MaxPool2d,
                    3: torch.nn.MaxPool3d,
                }[self.spatial_dims]
                # Calculate padding for each dimension
                padding = [k // 2 for k in self.proj_kernel_size]  # type: ignore
                self._dilator = pooler(
                    kernel_size=self.proj_kernel_size, stride=1, padding=padding
                )
            mask = self._dilator(mask.unsqueeze(1)).squeeze(1) > 0
        return mask

    def _forward_sga(self, layer, x, input_spacing, mask, input_grid_shapes, out):
        """
        Forward pass for Spatial Grouping Attention (SGA) layers.
        Also accumulates group/key relationships from past layers.
        """
        for s, (_x, _spacing, _mask, _grid_shape) in enumerate(
            zip(x, input_spacing, mask, input_grid_shapes)  # type: ignore
        ):
            _out = layer(_x, _spacing, _mask, _grid_shape)
            if len(out) > 0:
                # Accumulate group/key relationships
                _out["attn_q"] = _out["attn_q"] @ out[s]["attn_q"]
                _out["attn_k"] = _out["attn_k"] @ out[s]["attn_k"]
            out[s].update(_out)

        return out

    def forward(
        self,
        x: torch.Tensor | Sequence[torch.Tensor],
        input_spacing: Optional[Sequence[float] | Sequence[Sequence[float]]] = None,
        mask: Optional[torch.Tensor | Sequence[torch.Tensor]] = None,
    ) -> list[dict[str, torch.Tensor]]:
        x, input_spacing, mask, input_grid_shapes = self._prepare_inputs(
            x, input_spacing, mask
        )  # type: ignore
        num_ims = len(x)

        for i, _x in enumerate(x):
            x[i] = self.init_proj(_x)  # x --> [B, C, *spatial_dims]
            x[i] = x[i].flatten(start_dim=2).transpose(1, 2)  # [B, N, C]

        out = []
        for layer_idx in range(self.num_blocks):
            # Spatial Grouping Attention(s)
            out = self._forward_sga(
                self.sga_layers[layer_idx],
                x,
                input_spacing,
                mask,
                input_grid_shapes,
                out,
            )

            # Only do masking on first layer
            if mask is not None and layer_idx == 0:
                # Add masks to output
                for s in range(num_ims):
                    out[s]["mask"] = mask[s]

                # Then erase
                mask = None

            # Multi-Resolution Deformable Attention
            # TODO: Uncomment when MRDA is available
            # x = self.mrda_layers[layer_idx](x, input_spacing)

        return out

    def __repr__(self) -> str:
        """Return string representation of the object."""
        return f"ResolutionAwareTransformer{self.spatial_dims}D"


def init_weights(module):
    """
    Initialize the weights of a transformer model.

    This function applies Xavier/Glorot initialization to linear layers
    and normal initialization to embeddings, following common practices
    in transformer models.

    Args:
        module: The module to initialize
    """
    if isinstance(module, torch.nn.Linear):
        # Initialize linear layers with Xavier uniform initialization
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            torch.nn.init.constant_(module.bias, 0.0)

    elif isinstance(module, torch.nn.Embedding):
        # Initialize embeddings with normal distribution
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    elif isinstance(module, torch.nn.LayerNorm):
        # Initialize layer norm weights to 1 and bias to 0
        torch.nn.init.constant_(module.weight, 1.0)
        torch.nn.init.constant_(module.bias, 0.0)

    elif isinstance(module, torch.nn.MultiheadAttention):
        # Initialize multihead attention parameters
        if hasattr(module, "in_proj_weight") and module.in_proj_weight is not None:
            torch.nn.init.xavier_uniform_(module.in_proj_weight)
        if hasattr(module, "out_proj"):
            torch.nn.init.xavier_uniform_(module.out_proj.weight)
            if module.out_proj.bias is not None:
                torch.nn.init.constant_(module.out_proj.bias, 0.0)


def init_transformer_weights(model, std=0.02):
    """
    Alternative initialization function with configurable standard deviation.
    This follows the initialization scheme used in GPT and BERT models.

    Args:
        model: The transformer model to initialize
        std: Standard deviation for normal initialization (default: 0.02)
    """

    def _init_weights(module):
        if isinstance(module, torch.nn.Linear):
            # Truncated normal initialization
            torch.nn.init.trunc_normal_(
                module.weight, mean=0.0, std=std, a=-2 * std, b=2 * std
            )
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.trunc_normal_(
                module.weight, mean=0.0, std=std, a=-2 * std, b=2 * std
            )
        elif isinstance(module, torch.nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    model.apply(_init_weights)
