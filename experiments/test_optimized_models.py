#!/usr/bin/env python3
"""
Test script to verify model parameter counts for optimized configurations
"""

import os
import sys
import torch
from pathlib import Path
import yaml

# Add common utilities to path
sys.path.append(str(Path(__file__).parent / "common"))


def estimate_model_parameters(config):
    """Estimate model parameters from configuration"""
    model_config = config["model"]

    feature_dims = model_config.get("feature_dims", 256)
    num_blocks = model_config.get("num_blocks", 4)
    num_heads = model_config.get("num_heads", 16)
    mlp_ratio = model_config.get("mlp_ratio", 4)

    # Rough parameter estimation for ResolutionAwareTransformer
    # This is approximate - actual count depends on implementation details

    # Input projection layer
    input_proj = 3 * feature_dims  # Assuming 3 input channels

    # Embedding and positional encoding
    embeddings = feature_dims * 1000  # Rough estimate for positional embeddings

    # Per-block parameters
    per_block_params = 0

    # Attention parameters per block
    # Q, K, V projections: 3 * feature_dims * feature_dims
    attention_params = 3 * feature_dims * feature_dims
    # Output projection: feature_dims * feature_dims
    attention_params += feature_dims * feature_dims
    # Layer norm: 2 * feature_dims (weight + bias)
    attention_params += 2 * feature_dims

    # MLP parameters per block
    mlp_hidden = feature_dims * mlp_ratio
    # First linear layer: feature_dims * mlp_hidden
    mlp_params = feature_dims * mlp_hidden
    # Second linear layer: mlp_hidden * feature_dims
    mlp_params += mlp_hidden * feature_dims
    # Layer norm: 2 * feature_dims
    mlp_params += 2 * feature_dims

    per_block_params = attention_params + mlp_params

    # Total transformer parameters
    transformer_params = input_proj + embeddings + (num_blocks * per_block_params)

    # Segmentation head (approximate)
    seg_head_params = feature_dims * 128 + 128 * 64 + 64 * 1  # Rough estimate

    total_params = transformer_params + seg_head_params

    return {
        "input_projection": input_proj,
        "embeddings": embeddings,
        "per_block": per_block_params,
        "total_transformer": transformer_params,
        "segmentation_head": seg_head_params,
        "total": total_params,
    }


def test_model_creation(config_path):
    """Test actual model creation and parameter counting"""
    from models import create_model

    print(f"\nTesting model creation with: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Estimate parameters
    estimated = estimate_model_parameters(config)
    print(f"Estimated parameters: {estimated['total']:,}")

    # Create actual model
    try:
        model_config = config["model"].copy()
        model_name = model_config.pop("name")

        # Handle attention_type list (convert to first type for creation)
        attention_type = model_config.get("attention_type", "dense")
        if isinstance(attention_type, list):
            # For now, use the most common type in the list for estimation
            dense_count = attention_type.count("dense")
            sparse_count = attention_type.count("sparse")
            model_config["attention_type"] = (
                "dense" if dense_count >= sparse_count else "sparse"
            )

        model = create_model(
            model_name="rat", task="segmentation", num_classes=1, **model_config
        )

        # Count actual parameters
        actual_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Actual parameters: {actual_params:,}")

        # Compare to baseline (~19M)
        baseline_params = 19_000_000
        ratio = actual_params / baseline_params
        print(f"Ratio to baseline (19M): {ratio:.2f}x")

        if 0.8 <= ratio <= 1.5:  # Within reasonable range
            print("✅ Parameter count is within target range")
        else:
            print("⚠️ Parameter count is outside target range")

        return actual_params

    except Exception as e:
        print(f"❌ Error creating model: {e}")
        return None


def main():
    print("=" * 60)
    print("Model Parameter Analysis for Optimized Configurations")
    print("=" * 60)

    # Test optimized medical segmentation config
    medseg_config = (
        Path(__file__).parent / "medical_segmentation/configs/rat_optimized.yaml"
    )
    if medseg_config.exists():
        test_model_creation(medseg_config)
    else:
        print(f"Config not found: {medseg_config}")

    # Test baseline for comparison
    baseline_config = (
        Path(__file__).parent / "medical_segmentation/configs/rat_multiscale.yaml"
    )
    if baseline_config.exists():
        test_model_creation(baseline_config)
    else:
        print(f"Baseline config not found: {baseline_config}")

    print("\n" + "=" * 60)
    print("Parameter Analysis Complete")
    print("Target: ~19M parameters (matching baseline model)")
    print("Recommendation: Use feature_dims=384, num_blocks=6, num_heads=12")
    print("Memory optimization: Sparse attention for first 2 blocks, dense for rest")
    print("=" * 60)


if __name__ == "__main__":
    main()
