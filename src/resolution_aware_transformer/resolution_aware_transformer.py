"""Core functionality for resolution_aware_transformer."""

from typing import Literal, Optional, Sequence, Type

# from multi_res_mossa import MultiResMoSSA
from RoSE import MultiRes_RoSE_Block
from spatial_grouping_attention import (
    DenseSpatialGroupingAttention,
    SparseSpatialGroupingAttention,
)
from spatial_grouping_attention.utils import to_tuple
import torch


class ResolutionAwareTransformer(torch.nn.Module):
    """Resolution Aware Transformer class
    Args:
        spatial_dims: The spatial dimensions of the input images. (e.g. 2 or 3)
        input_features: Number of input features (e.g., channels in an image)
        proj_kernel_size: Kernel size for the initial projection layer. Default is 7.
        proj_padding: Padding for the initial projection layer. Default is "same".
        feature_dims: The dimensionality of the embedding space.
        num_blocks: The number of spatial grouping attention and
            mixture of sparse spatial attention pairs
        sga_attention_type: The type of spatial grouping layer attention
            (default is "dense"). Options are "dense" or "sparse"
        sga_kernel_size: Size of the convolutional kernel for strided convolutions
            producing the lower res embeddings in each spatial grouping attention layer
        stride: Stride for the spatial grouping attention layer strided convolutions
            (default is half kernel size)
        num_heads: Number of attention heads
        iters: Number of iterations for the spatial grouping attention mechanism
            (default is 3)
        mlp_ratio: Ratio of hidden to input/output dimensions in MLPs
            (default is 4)
        mlp_dropout: Dropout rate for MLPs (default is 0.0)
        mlp_bias: Whether to use bias in MLPs (default is True)
        mlp_activation: Activation function for MLPs (default is GELU)
        qkv_bias: Whether to use bias in the query/key/value linear layers
            (default is True)
        base_theta: Base theta value for the rotary position embedding
            (default is 1e4)
        learnable_rose: Whether to use learnable rotary spatial embeddings
            (default is True)
        learnable_rose_scaling: Whether to use learnable scaling for the rotary spatial embeddings (default is True).
            If True, the model learns a scaling factor for each spatial dimension,
            allowing the rotary spatial embeddings to adapt to the data. This can
            improve performance when the spatial frequency content varies across
            datasets or tasks. Set to False to use fixed scaling.
        rose_log_scaling: Whether to use log scaling for the rotary spatial embeddings
            (default is True).
            If True, the scaling factors for the rotary spatial embeddings are
            parameterized in log-space, which can improve stability and allow for
            a wider dynamic range. Set to False to use linear scaling.
        init_jitter_std: Standard deviation for initial jitter in the rotary
            embeddings (default is 0.02).
            Controls the amount of random noise added to the initial rotary
            spatial embeddings. Increasing this value can help with exploration
            during training, but may make convergence slower.
        rotary_ratio: Fraction of the feature dimension to rotate
        frequency_scaling: Frequency scaling method for the rotary position embedding
            (default is "sqrt")
        leading_tokens: Number of context tokens to prepend to input sequences.
            Useful for CLS or buffer tokens, for example. These tokens are not
            rotated during rotational position embedding and not included in spatial
            grouping attention.
        spacing: Default real-world pixel spacing for the input data
            (default is None, which uses a default spacing of 1.0 for all dimensions).
            Can be specified at initialization or passed during the forward pass.
    """

    def __init__(
        self,
        spatial_dims: int = 2,
        input_features: int = 3,
        proj_kernel_size: int | Sequence[int] = 7,
        proj_padding: int | Sequence[int] | Literal["same"] = "same",
        feature_dims: int = 128,
        num_blocks: int = 4,
        sga_attention_type: str | Sequence[str] = "dense",
        sga_kernel_size: int | Sequence[int] = 7,
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
        learnable_rose_scaling: bool = True,
        rose_log_scaling: bool = True,
        init_jitter_std: float = 0.02,
        rotary_ratio: float = 0.5,
        frequency_scaling: str = "sqrt",
        leading_tokens: int = 0,
        spacing: Optional[float | Sequence[float]] = None,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.input_features = input_features
        self.proj_kernel_size = to_tuple(
            proj_kernel_size, spatial_dims, dtype_caster=int, allow_nested=False
        )
        if proj_padding == "same":
            self.proj_padding = tuple(
                k // 2 for k in self.proj_kernel_size  # type: ignore
            )
        else:
            self.proj_padding = to_tuple(
                proj_padding, spatial_dims, dtype_caster=int, allow_nested=False
            )
        self.feature_dims = feature_dims
        self.num_blocks = num_blocks
        self.sga_attention_type = to_tuple(
            sga_attention_type, num_blocks, allow_nested=False
        )
        self.sga_kernel_size = to_tuple(
            sga_kernel_size, spatial_dims, dtype_caster=int, allow_nested=False
        )
        if stride is None:
            # Default stride is half kernel size, minimum 1
            self.stride = tuple(
                max(1, k // 2) for k in self.sga_kernel_size  # type: ignore
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
        self.rotary_ratio = rotary_ratio
        self.frequency_scaling = frequency_scaling
        self.learnable_rose_scaling = learnable_rose_scaling
        self.rose_log_scaling = rose_log_scaling
        if leading_tokens > 0:
            self.leading_tokens = torch.nn.Parameter(
                torch.rand((1, leading_tokens, feature_dims))
            )
        else:
            self.leading_tokens = None
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
        mr_attn_layers = []
        sga_kwargs = {
            "feature_dims": feature_dims,
            "spatial_dims": spatial_dims,
            "kernel_size": self.sga_kernel_size,
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
            "rotary_ratio": rotary_ratio,
            "frequency_scaling": frequency_scaling,
            "learnable_rose_scaling": learnable_rose_scaling,
            "rose_log_scaling": rose_log_scaling,
            "spacing": self._default_spacing,
        }
        mr_attn_kwargs = {
            "feature_dims": feature_dims,
            "num_heads": num_heads,
            "spatial_dims": spatial_dims,
            "mlp_ratio": mlp_ratio,
            "mlp_dropout": mlp_dropout,
            "mlp_bias": mlp_bias,
            "mlp_activation": mlp_activation,
            "qkv_bias": qkv_bias,
            "base_theta": base_theta,
            "learnable": learnable_rose,
            "init_jitter_std": init_jitter_std,
            "frequency_scaling": frequency_scaling,
            "learnable_scale": learnable_rose_scaling,
            "log_scale": rose_log_scaling,
            "rotary_ratio": rotary_ratio,
        }
        for n in range(num_blocks):
            sga = (
                DenseSpatialGroupingAttention
                if self.sga_attention_type[n] == "dense"
                else SparseSpatialGroupingAttention
            )
            sga_layers.append(sga(**sga_kwargs))
            mr_attn_layers.append(MultiRes_RoSE_Block(**mr_attn_kwargs))

        self.sga_layers = torch.nn.ModuleList(sga_layers)
        self.mr_attn_layers = torch.nn.ModuleList(mr_attn_layers)

        # Initialize weights
        self.apply(init_transformer_weights)

    def _prepare_inputs(self, x, input_spacing, mask):
        # Make everything lists if not already (for image res pyramid)
        if isinstance(x, torch.Tensor):
            x = [x]
        assert isinstance(
            x, Sequence
        ), "Input x must be a tensor or a sequence of tensors"
        num_ims = len(x)

        if isinstance(mask, torch.Tensor):
            mask = [mask]

        if input_spacing is None:
            input_spacing = [
                self._default_spacing,
            ] * num_ims
        input_spacing = to_tuple(input_spacing, self.spatial_dims)
        if not isinstance(input_spacing[0], Sequence):
            input_spacing = [input_spacing]

        # Get input grid shapes
        input_grid_shapes = [tuple(_x.shape[-self.spatial_dims :]) for _x in x]

        # Dilate masks if necessary to handle init_proj kernel size
        if mask is not None:
            for i, m in enumerate(mask):
                mask[i] = self.dilate_mask(m)
        else:
            mask = [None] * num_ims

        assert len(mask) == num_ims, "Mask length must match number of input images"
        assert (
            len(input_spacing) == num_ims
        ), "Spacing length must match number of input images"
        assert (
            len(input_grid_shapes) == num_ims
        ), "Grid shape length must match number of input images"

        out = [
            {
                "x_out": _x,
                "out_spacing": _spacing,
                "out_grid_shape": _shape,
                "mask": _mask,
            }
            for _x, _spacing, _shape, _mask in zip(
                x, input_spacing, input_grid_shapes, mask
            )  # type: ignore
        ]
        return out

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
            mask = self._dilator(mask.float().unsqueeze(1)).squeeze(1) > 0
        return mask

    def _forward_sga(self, layer, out):
        """
        Forward pass for Spatial Grouping Attention (SGA) layers.
        Also accumulates group/key relationships from past layers.
        """
        for s in range(len(out)):
            _out = layer(
                out[s]["x_out"],
                out[s]["out_spacing"],
                out[s]["out_grid_shape"],
                out[s]["mask"],
            )
            if "attn_q" in out[s]:
                # Accumulate group/key relationships only if they exist
                _out["attn_q"] = _out["attn_q"] @ out[s]["attn_q"]
                _out["attn_k"] = _out["attn_k"] @ out[s]["attn_k"]

            _out["mask"] = None  # Remove mask for next layers
            out[s].update(_out)

        return out

    def _forward_mr_attn(self, layer, out):
        """
        Forward pass for Multi-Resolution Attention layers.
        """
        has_leader = False
        x = []
        spacings = []
        grid_shapes = []

        for _out in out:
            if "x_out" in _out:
                x += [_out["x_out"]]
                spacings += [_out["out_spacing"]]
                grid_shapes += [_out["out_grid_shape"]]
            elif "leading_tokens" in _out:
                x = [_out["leading"]] + x
                has_leader = True
            else:
                raise ValueError(f"Unexpected entry in outputs dictionary:\n\t{_out}")

        x = layer(x, spacings, grid_shapes)

        for i, _x in enumerate(x):
            if i == 0 and has_leader:
                out[i]["leading_tokens"] = _x
            else:
                out[i]["x_out"] = _x

        return out

    def forward(
        self,
        x: torch.Tensor | Sequence[torch.Tensor],
        input_spacing: Optional[Sequence[float] | Sequence[Sequence[float]]] = None,
        mask: Optional[torch.Tensor | Sequence[torch.Tensor]] = None,
    ) -> list[dict[str, torch.Tensor]]:
        out = self._prepare_inputs(x, input_spacing, mask)  # type: ignore

        for i, _out in enumerate(out):
            if "x_out" not in _out:
                continue
            _x = _out["x_out"]
            _x = self.init_proj(_x)  # x --> [B, C, *spatial_dims]
            out[i]["x_out"] = _x.flatten(2).transpose(1, 2)  # [B, N, C]

        for layer_idx in range(self.num_blocks):
            # Spatial Grouping Attention(s)
            out = self._forward_sga(
                self.sga_layers[layer_idx],
                out,
            )

            out = self._forward_mr_attn(
                self.mr_attn_layers[layer_idx],
                out,
            )

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
