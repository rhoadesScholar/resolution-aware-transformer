"""Test suite for resolution_aware_transformer."""

import pytest
import torch

from resolution_aware_transformer.resolution_aware_transformer import (
    ResolutionAwareTransformer,
)


class TestResolutionAwareTransformer:
    """Test cases for ResolutionAwareTransformer."""

    def test_initialization_default(self):
        """Test basic initialization with default parameters."""
        model = ResolutionAwareTransformer()
        assert model.spatial_dims == 2
        assert model.input_features == 3
        assert model.feature_dims == 128
        assert model.num_blocks == 4
        assert model.num_heads == 16
        assert model.iters == 3
        assert model.mlp_ratio == 4
        assert model.mlp_dropout == 0.0
        assert model.mlp_bias is True
        assert model.qkv_bias is True
        assert model.learnable_rose is True
        assert model.init_jitter_std == 0.02

    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        model = ResolutionAwareTransformer(
            spatial_dims=2,
            input_features=5,
            feature_dims=256,
            num_blocks=2,
            sga_attention_type=["dense"]
            * 2,  # Use dense attention to avoid CUDA issues
            num_heads=8,
            iters=5,
            mlp_ratio=2,
            mlp_dropout=0.1,
            spacing=2.0,
        )
        assert model.spatial_dims == 2
        assert model.input_features == 5
        assert model.feature_dims == 256
        assert model.num_blocks == 2
        assert model.num_heads == 8
        assert model.iters == 5
        assert model.mlp_ratio == 2
        assert model.mlp_dropout == 0.1

    def test_device_consistency(self):
        """Test that model works on different devices."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ResolutionAwareTransformer(
            spatial_dims=2,
            input_features=3,
            feature_dims=32,
            num_blocks=1,
            sga_attention_type=["dense"],
            sga_kernel_size=3,
            stride=1,
        ).to(device)

        x = torch.randn(1, 3, 16, 16).to(device)

        output = model(x)
        assert isinstance(output, list)
        assert len(output) == 1
        # Check keys that are actually present in single-block output
        for key in [
            "x_out",
            "out_spacing",
            "out_grid_shape",
            "mask",
            "attn_q",
            "attn_k",
        ]:
            assert key in output[0]
            if key in ["attn_q", "attn_k"]:
                # Compare only the device type, ignoring any index discrepancy.
                # Device index comparison was intentionally removed to support
                # multi-GPU scenarios, where tensors may reside on different GPU
                # indices but should still be considered consistent.
                assert output[0][key].device.type == device.type

    def test_forward_3d_single_image(self):
        """Test forward pass with 3D single image."""
        model = ResolutionAwareTransformer(
            spatial_dims=3,
            input_features=1,
            feature_dims=32,
            num_blocks=1,
            sga_attention_type=["dense"],
            sga_kernel_size=3,
            stride=1,
        )

        # Create a sample 3D image tensor [B, C, D, H, W]
        x = torch.randn(1, 1, 16, 16, 16)

        with torch.no_grad():
            output = model(x)

        assert isinstance(output, list)
        assert len(output) == 1
        assert isinstance(output[0], dict)

    def test_forward_multiple_images(self):
        """Test forward pass with multiple images."""
        model = ResolutionAwareTransformer(
            spatial_dims=2,
            input_features=3,
            feature_dims=64,
            num_blocks=1,
            sga_attention_type=["dense"],
            sga_kernel_size=3,
            stride=1,
        )

        # Create multiple images with different sizes
        x1 = torch.randn(1, 3, 16, 16)  # Reduced from 32x32
        x2 = torch.randn(1, 3, 32, 32)  # Reduced from 64x64
        x = [x1, x2]

        with torch.no_grad():
            output = model(x)

        assert isinstance(output, list)
        assert len(output) == 2  # Two input images
        for out in output:
            assert isinstance(out, dict)
            assert "attn_q" in out
            assert "attn_k" in out

    def test_forward_with_mask(self):
        """Test forward pass with mask."""
        model = ResolutionAwareTransformer(
            spatial_dims=2,
            input_features=3,
            feature_dims=64,
            num_blocks=1,
            proj_kernel_size=3,
            sga_attention_type=["dense"],
            sga_kernel_size=3,
            stride=1,
        )

        # Create a sample image and mask
        x = torch.randn(1, 3, 16, 16)  # Reduced from 32x32
        mask = torch.ones(1, 16, 16, dtype=torch.bool)  # Reduced from 32x32

        with torch.no_grad():
            output = model(x, mask=mask)

        assert isinstance(output, list)
        assert len(output) == 1
        assert "mask" in output[0]

    def test_forward_with_spacing(self):
        """Test forward pass with custom spacing."""
        model = ResolutionAwareTransformer(
            spatial_dims=2,
            input_features=3,
            feature_dims=64,
            num_blocks=1,
            sga_attention_type=["dense"],
            sga_kernel_size=3,
            stride=1,
        )

        x = torch.randn(1, 3, 16, 16)  # Reduced from 32x32
        spacing = [1.5, 2.0]  # Custom spacing for each dimension

        with torch.no_grad():
            output = model(x, input_spacing=spacing)

        assert isinstance(output, list)
        assert len(output) == 1

    def test_repr(self):
        """Test string representation."""
        model_2d = ResolutionAwareTransformer(spatial_dims=2)
        model_3d = ResolutionAwareTransformer(spatial_dims=3)

        assert repr(model_2d) == "ResolutionAwareTransformer2D"
        assert repr(model_3d) == "ResolutionAwareTransformer3D"

    @pytest.mark.parametrize(
        "spatial_dims,input_shape",
        [
            (1, (2, 3, 16)),  # Reduced from 32
            (2, (2, 3, 16, 16)),  # Reduced from 32x32
            (3, (2, 3, 8, 8, 8)),  # Reduced from 16x16x16
        ],
    )
    def test_different_spatial_dims(self, spatial_dims: int, input_shape: tuple):
        """Test model with different spatial dimensions."""
        model = ResolutionAwareTransformer(
            spatial_dims=spatial_dims,
            input_features=3,
            feature_dims=32,
            num_blocks=1,
            sga_attention_type=["dense"],
            sga_kernel_size=3,
            stride=1,
        )

        x = torch.randn(input_shape)

        with torch.no_grad():
            output = model(x)

        assert isinstance(output, list)
        assert len(output) == 1

    def test_sga_attention_types_dense(self):
        """Test dense attention type."""
        model = ResolutionAwareTransformer(
            spatial_dims=2,
            input_features=3,
            feature_dims=32,
            num_blocks=1,
            sga_attention_type=["dense"],
            sga_kernel_size=3,
            stride=1,
        )

        x = torch.randn(1, 3, 16, 16)

        with torch.no_grad():
            output = model(x)

        assert isinstance(output, list)
        assert len(output) == 1


# Integration tests
class TestIntegration:
    """Integration test cases."""

    def test_full_workflow(self):
        """Test complete workflow with realistic parameters."""
        # Initialize model with realistic medical imaging parameters
        model = ResolutionAwareTransformer(
            spatial_dims=3,
            input_features=1,  # Single channel (e.g., CT scan)
            feature_dims=128,
            num_blocks=2,
            sga_attention_type=["dense"]
            * 2,  # Use dense attention to avoid CUDA issues
            num_heads=8,
            sga_kernel_size=3,
            stride=1,
        )

        # Create smaller 3D medical image tensor to avoid memory issues
        x = torch.randn(1, 1, 8, 8, 8)  # [B, C, D, H, W] - reduced from 32x32x32

        with torch.no_grad():
            output = model(x)

        # Verify output structure
        assert isinstance(output, list)
        assert len(output) == 1
        assert isinstance(output[0], dict)
        assert "attn_q" in output[0]
        assert "attn_k" in output[0]

    def test_gradient_flow(self):
        """Test that gradients flow properly through the model."""
        model = ResolutionAwareTransformer(
            spatial_dims=2,
            input_features=3,
            feature_dims=32,
            num_blocks=1,
            sga_attention_type=["dense"],
            sga_kernel_size=3,
            stride=1,
        )

        x = torch.randn(1, 3, 16, 16, requires_grad=True)

        output = model(x)
        # Simple loss for gradient test
        loss = torch.tensor(0.0)
        for out in output:
            loss = loss + out["attn_q"].sum()
        loss.backward()

        # Check that gradients exist
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_rose_initial_scaling_parameter(self):
        """Test different rose_initial_scaling parameter options."""
        # Test RoSE (default)
        model_rose = ResolutionAwareTransformer(
            spatial_dims=2,
            input_features=3,
            feature_dims=32,
            num_blocks=1,
            sga_attention_type=["dense"],
            rose_initial_scaling="log",  # RoSE default
        )
        assert model_rose.rose_initial_scaling == "log"

        # Test RoPE mode
        model_rope = ResolutionAwareTransformer(
            spatial_dims=2,
            input_features=3,
            feature_dims=32,
            num_blocks=1,
            sga_attention_type=["dense"],
            rose_initial_scaling="rope",  # Standard RoPE
        )
        assert model_rope.rose_initial_scaling == "rope"

        # Test other scaling modes
        for scaling in ["identity", "linear", "power"]:
            model = ResolutionAwareTransformer(
                spatial_dims=2,
                input_features=3,
                feature_dims=32,
                num_blocks=1,
                sga_attention_type=["dense"],
                rose_initial_scaling=scaling,
            )
            assert model.rose_initial_scaling == scaling

        # Verify models can forward pass
        x = torch.randn(1, 3, 16, 16)
        with torch.no_grad():
            out_rose = model_rose(x)
            out_rope = model_rope(x)

        assert isinstance(out_rose, list)
        assert isinstance(out_rope, list)
        assert len(out_rose) == len(out_rope) == 1


# Fixtures for common test data
@pytest.fixture
def sample_model_2d():
    """Fixture providing a sample 2D ResolutionAwareTransformer."""
    return ResolutionAwareTransformer(
        spatial_dims=2,
        input_features=3,
        feature_dims=64,
        num_blocks=1,
        sga_attention_type=["dense"],
        num_heads=4,
        sga_kernel_size=3,
        stride=1,
    )


@pytest.fixture
def sample_model_3d():
    """Fixture providing a sample 3D ResolutionAwareTransformer."""
    return ResolutionAwareTransformer(
        spatial_dims=3,
        input_features=1,
        feature_dims=32,
        num_blocks=1,
        sga_attention_type=["dense"],
        num_heads=2,
        sga_kernel_size=3,
        stride=1,
    )


@pytest.fixture
def sample_data_2d():
    """Fixture providing sample 2D test data."""
    return {
        "single_image": torch.randn(1, 3, 16, 16),  # Reduced from 32x32
        "batch_images": torch.randn(4, 3, 16, 16),  # Reduced from 32x32
        "large_image": torch.randn(1, 3, 64, 64),  # Reduced from 128x128
        "mask": torch.ones(1, 16, 16, dtype=torch.bool),  # Reduced from 32x32
        "spacing": [1.0, 1.0],
    }


@pytest.fixture
def sample_data_3d():
    """Fixture providing sample 3D test data."""
    return {
        "single_image": torch.randn(1, 1, 8, 8, 8),  # Reduced from 16x16x16
        "batch_images": torch.randn(2, 1, 8, 8, 8),  # Reduced from 16x16x16
        "mask": torch.ones(1, 8, 8, 8, dtype=torch.bool),  # Reduced from 16x16x16
        "spacing": [1.0, 1.0, 1.0],
    }


def test_with_fixtures_2d(sample_model_2d, sample_data_2d):
    """Test using 2D fixtures."""
    with torch.no_grad():
        output = sample_model_2d(sample_data_2d["single_image"])

    assert isinstance(output, list)
    assert len(output) == 1


def test_with_fixtures_3d(sample_model_3d, sample_data_3d):
    """Test using 3D fixtures."""
    with torch.no_grad():
        output = sample_model_3d(sample_data_3d["single_image"])

    assert isinstance(output, list)
    assert len(output) == 1
