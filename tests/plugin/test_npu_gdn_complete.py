# Copyright (c) 2025, BAAI. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Tests for GDN NPU implementation."""

import pytest
import torch


@pytest.fixture
def device():
    """Get NPU device if available."""
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    return torch.device("npu:0")


@pytest.fixture
def gdn_manager():
    """Get OpManager with GDN support."""
    try:
        from transformer_engine.plugin.core.manager import OpManager
        manager = OpManager()
        return manager
    except ImportError:
        pytest.skip("TransformerEngine Plugin not available")


class TestGDNForwardBackward:
    """Test forward and backward passes."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("batch,seq,heads,head_dim", [
        (2, 128, 8, 64),    # Normal scale
        (1, 64, 4, 32),     # Minimal scale
        (4, 256, 16, 128),  # Large scale
    ])
    def test_forward_backward(self, device, gdn_manager, dtype, batch, seq, heads, head_dim):
        """Test complete forward and backward pass."""
        # Create inputs with gradients
        query = torch.randn(batch, seq, heads, head_dim, dtype=dtype, device=device, requires_grad=True)
        key = torch.randn(batch, seq, heads, head_dim, dtype=dtype, device=device, requires_grad=True)
        value = torch.randn(batch, seq, heads, head_dim, dtype=dtype, device=device, requires_grad=True)
        g = torch.randn(batch, seq, heads, dtype=torch.float32, device=device, requires_grad=True)
        beta = torch.randn(batch, seq, heads, dtype=torch.float32, device=device, requires_grad=True)

        # Forward pass
        output, final_state = gdn_manager.call(
            "gated_delta_net_forward",
            query=query,
            key=key,
            value=value,
            g=g,
            beta=beta,
        )

        # Validate output
        assert output.shape == (batch, seq, heads, head_dim)
        assert output.dtype == dtype
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"

        # Backward pass
        grad_output = torch.randn_like(output)
        output.backward(grad_output)

        # Validate gradients
        for tensor, name in [
            (query, "query"),
            (key, "key"),
            (value, "value"),
            (g, "g"),
            (beta, "beta"),
        ]:
            assert tensor.grad is not None, f"{name} gradient is None"
            assert not torch.isnan(tensor.grad).any(), f"{name} gradient contains NaN"
            assert not torch.isinf(tensor.grad).any(), f"{name} gradient contains Inf"

        print(f"✓ Forward+backward test passed: dtype={dtype}, shape=[{batch},{seq},{heads},{head_dim}]")


class TestGDNShapes:
    """Test various input shapes and edge cases."""

    def test_non_chunk_aligned_sequence(self, device, gdn_manager):
        """Test sequence length that doesn't align with chunk_size."""
        dtype = torch.bfloat16
        batch, seq, heads, head_dim = 2, 63, 8, 64  # 63 is not divisible by 64
        chunk_size = 64

        query = torch.randn(batch, seq, heads, head_dim, dtype=dtype, device=device)
        key = torch.randn(batch, seq, heads, head_dim, dtype=dtype, device=device)
        value = torch.randn(batch, seq, heads, head_dim, dtype=dtype, device=device)
        g = torch.randn(batch, seq, heads, dtype=torch.float32, device=device)
        beta = torch.randn(batch, seq, heads, dtype=torch.float32, device=device)

        output, _ = gdn_manager.call(
            "gated_delta_net_forward",
            query=query,
            key=key,
            value=value,
            g=g,
            beta=beta,
            chunk_size=chunk_size,
        )

        assert output.shape == (batch, seq, heads, head_dim)
        assert not torch.isnan(output).any()
        print("✓ Non-chunk-aligned sequence test passed")

    @pytest.mark.parametrize("chunk_size", [16, 32, 64, 128])
    def test_different_chunk_sizes(self, device, gdn_manager, chunk_size):
        """Test different chunk sizes."""
        dtype = torch.bfloat16
        batch, seq, heads, head_dim = 2, 128, 8, 64

        query = torch.randn(batch, seq, heads, head_dim, dtype=dtype, device=device)
        key = torch.randn(batch, seq, heads, head_dim, dtype=dtype, device=device)
        value = torch.randn(batch, seq, heads, head_dim, dtype=dtype, device=device)
        g = torch.randn(batch, seq, heads, dtype=torch.float32, device=device)
        beta = torch.randn(batch, seq, heads, dtype=torch.float32, device=device)

        output, _ = gdn_manager.call(
            "gated_delta_net_forward",
            query=query,
            key=key,
            value=value,
            g=g,
            beta=beta,
            chunk_size=chunk_size,
        )

        assert output.shape == (batch, seq, heads, head_dim)
        assert not torch.isnan(output).any()
        print(f"✓ Chunk size {chunk_size} test passed")


class TestGDNNumericalStability:
    """Test numerical stability and precision."""

    def test_no_nan_with_large_values(self, device, gdn_manager):
        """Test that large input values don't cause NaN."""
        dtype = torch.bfloat16
        batch, seq, heads, head_dim = 2, 128, 8, 64

        # Create large values
        query = torch.randn(batch, seq, heads, head_dim, dtype=dtype, device=device) * 10
        key = torch.randn(batch, seq, heads, head_dim, dtype=dtype, device=device) * 10
        value = torch.randn(batch, seq, heads, head_dim, dtype=dtype, device=device) * 10
        g = torch.randn(batch, seq, heads, dtype=torch.float32, device=device) * 5
        beta = torch.randn(batch, seq, heads, dtype=torch.float32, device=device).abs() * 2

        output, _ = gdn_manager.call(
            "gated_delta_net_forward",
            query=query,
            key=key,
            value=value,
            g=g,
            beta=beta,
        )

        assert not torch.isnan(output).any(), "Output contains NaN with large values"
        assert not torch.isinf(output).any(), "Output contains Inf with large values"
        print("✓ Large values stability test passed")

    def test_gradient_numerical_stability(self, device, gdn_manager):
        """Test gradient computation doesn't produce NaN."""
        dtype = torch.bfloat16
        batch, seq, heads, head_dim = 2, 128, 8, 64

        query = torch.randn(batch, seq, heads, head_dim, dtype=dtype, device=device, requires_grad=True)
        key = torch.randn(batch, seq, heads, head_dim, dtype=dtype, device=device, requires_grad=True)
        value = torch.randn(batch, seq, heads, head_dim, dtype=dtype, device=device, requires_grad=True)
        g = torch.randn(batch, seq, heads, dtype=torch.float32, device=device, requires_grad=True)
        beta = torch.randn(batch, seq, heads, dtype=torch.float32, device=device, requires_grad=True).abs()

        output, _ = gdn_manager.call(
            "gated_delta_net_forward",
            query=query,
            key=key,
            value=value,
            g=g,
            beta=beta,
        )

        loss = output.sum()
        loss.backward()

        # Check gradients are finite
        for tensor, name in [(query, "query"), (key, "key"), (value, "value"), (g, "g"), (beta, "beta")]:
            grad = tensor.grad
            assert grad is not None, f"{name} gradient is None"
            finite_ratio = torch.isfinite(grad).float().mean().item()
            assert finite_ratio > 0.99, f"{name} gradient has {1-finite_ratio:.2%} non-finite values"

        print("✓ Gradient numerical stability test passed")


class TestGDNErrorHandling:
    """Test error handling and validation."""

    def test_shape_mismatch_error(self, device, gdn_manager):
        """Test that shape mismatches raise appropriate errors."""
        dtype = torch.bfloat16

        query = torch.randn(2, 128, 8, 64, dtype=dtype, device=device)
        key = torch.randn(2, 128, 8, 64, dtype=dtype, device=device)
        value = torch.randn(2, 128, 8, 32, dtype=dtype, device=device)  # Wrong head_dim
        g = torch.randn(2, 128, 8, dtype=torch.float32, device=device)
        beta = torch.randn(2, 128, 8, dtype=torch.float32, device=device)

        with pytest.raises((ValueError, RuntimeError)):
            gdn_manager.call(
                "gated_delta_net_forward",
                query=query,
                key=key,
                value=value,
                g=g,
                beta=beta,
            )
        print("✓ Shape mismatch error test passed")

    def test_invalid_chunk_size_error(self, device, gdn_manager):
        """Test that invalid chunk_size raises error."""
        dtype = torch.bfloat16
        batch, seq, heads, head_dim = 2, 128, 8, 64

        query = torch.randn(batch, seq, heads, head_dim, dtype=dtype, device=device)
        key = torch.randn(batch, seq, heads, head_dim, dtype=dtype, device=device)
        value = torch.randn(batch, seq, heads, head_dim, dtype=dtype, device=device)
        g = torch.randn(batch, seq, heads, dtype=torch.float32, device=device)
        beta = torch.randn(batch, seq, heads, dtype=torch.float32, device=device)

        # Chunk size must be power of 2
        with pytest.raises((ValueError, RuntimeError)):
            gdn_manager.call(
                "gated_delta_net_forward",
                query=query,
                key=key,
                value=value,
                g=g,
                beta=beta,
                chunk_size=63,  # Not a power of 2
            )
        print("✓ Invalid chunk_size error test passed")


def test_gdn_availability():
    """Test GDN availability check."""
    from transformer_engine.plugin.core.backends.vendor.npu.gated_delta_net import (
        is_gated_delta_net_available,
    )

    available = is_gated_delta_net_available()
    print(f"GDN available: {available}")

    if not available:
        pytest.skip("GDN operators not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
