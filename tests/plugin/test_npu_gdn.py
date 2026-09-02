# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""
Unit tests for NPU GDN (Gated Delta Net) implementation.

Tests follow the requirements from FlagOS训练全流程适配实施手册.md Section 7.5:
- Multiple dtypes (FP32, BF16, FP16)
- Various shapes (normal, minimal, edge cases)
- Contiguous and non-contiguous inputs
- Forward numerical correctness
- Comparison with Reference implementation
"""

import pytest
import torch

# Skip all tests if NPU is not available
try:
    import torch_npu
    NPU_AVAILABLE = torch.npu.is_available() and torch.npu.device_count() > 0
except (ImportError, AttributeError):
    NPU_AVAILABLE = False

try:
    from transformer_engine.plugin.core.manager import OpManager
    TE_PLUGIN_AVAILABLE = True
except ImportError:
    TE_PLUGIN_AVAILABLE = False


@pytest.mark.skipif(not NPU_AVAILABLE, reason="NPU not available")
@pytest.mark.skipif(not TE_PLUGIN_AVAILABLE, reason="TE Plugin not available")
class TestNPUGatedDeltaNet:
    """Test suite for NPU GDN implementation following handbook Section 7.5"""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    @pytest.mark.parametrize(
        "batch,seq,heads,head_dim,value_dim",
        [
            (2, 128, 8, 64, 64),   # Normal shape
            (1, 64, 4, 32, 32),    # Minimal shape
            (4, 256, 16, 128, 128), # Larger shape
            (1, 63, 4, 32, 32),    # Non-chunk-aligned sequence
        ],
    )
    def test_gdn_forward_dtype_and_shape(self, dtype, batch, seq, heads, head_dim, value_dim):
        """
        Test GDN forward with various dtypes and shapes.

        Requirements from handbook 7.5:
        - FP32, BF16, FP16
        - Normal shape, minimal shape, non-divisible shape
        """
        device = torch.device("npu:0")

        # Create inputs
        query = torch.randn(batch, seq, heads, head_dim, dtype=dtype, device=device)
        key = torch.randn(batch, seq, heads, head_dim, dtype=dtype, device=device)
        value = torch.randn(batch, seq, heads, value_dim, dtype=dtype, device=device)
        g = torch.randn(batch, seq, heads, dtype=dtype, device=device)
        beta = torch.randn(batch, seq, heads, dtype=dtype, device=device)

        manager = OpManager()

        # Test NPU Vendor implementation
        out_vendor, state_vendor = manager.call(
            "gated_delta_net_forward",
            query=query, key=key, value=value, g=g, beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm=False,
        )

        # Verify output properties
        assert out_vendor.shape == (batch, seq, heads, value_dim)
        assert out_vendor.dtype == dtype
        assert out_vendor.device == device
        assert out_vendor.is_contiguous()
        assert not torch.isnan(out_vendor).any()
        assert not torch.isinf(out_vendor).any()

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_gdn_vendor_vs_reference(self, dtype):
        """
        Test NPU Vendor implementation against Reference implementation.

        Requirements from handbook 7.5 and 13.1:
        - Compare with Reference
        - Max absolute/relative error
        - Precision: atol=1e-2, rtol=2e-3
        """
        device = torch.device("npu:0")
        batch, seq, heads, head_dim, value_dim = 2, 128, 8, 64, 64

        # Create inputs
        query = torch.randn(batch, seq, heads, head_dim, dtype=dtype, device=device)
        key = torch.randn(batch, seq, heads, head_dim, dtype=dtype, device=device)
        value = torch.randn(batch, seq, heads, value_dim, dtype=dtype, device=device)
        g = torch.randn(batch, seq, heads, dtype=dtype, device=device)
        beta = torch.randn(batch, seq, heads, dtype=dtype, device=device)

        manager = OpManager()

        # Get Vendor implementation (NPU AscendC)
        try:
            out_vendor, _ = manager.call(
                "gated_delta_net_forward",
                query=query, key=key, value=value, g=g, beta=beta,
                prefer="vendor:NPU"
            )
        except Exception as e:
            pytest.skip(f"Vendor implementation not available: {e}")

        # Get Reference implementation (PyTorch)
        out_ref, _ = manager.call(
            "gated_delta_net_forward",
            query=query, key=key, value=value, g=g, beta=beta,
            prefer="reference"
        )

        # Calculate errors
        abs_error = (out_vendor - out_ref).abs()
        max_abs_error = abs_error.max().item()

        rel_error = abs_error / (out_ref.abs() + 1e-8)
        max_rel_error = rel_error.max().item()

        mean_rel_error = rel_error.mean().item()

        # Print diagnostics
        print(f"\nDtype: {dtype}")
        print(f"Max absolute error: {max_abs_error:.6e}")
        print(f"Max relative error: {max_rel_error:.6e}")
        print(f"Mean relative error: {mean_rel_error:.6e}")

        # Verify precision (handbook 13.1 standards)
        assert torch.allclose(out_vendor, out_ref, atol=1e-2, rtol=2e-3), (
            f"Vendor vs Reference mismatch: "
            f"max_abs_error={max_abs_error:.6e}, "
            f"max_rel_error={max_rel_error:.6e}"
        )

    def test_gdn_non_contiguous_input(self):
        """
        Test with non-contiguous inputs.

        Requirements from handbook 7.5:
        - Contiguous and non-contiguous inputs
        """
        device = torch.device("npu:0")
        dtype = torch.float32
        batch, seq, heads, head_dim, value_dim = 2, 128, 8, 64, 64

        # Create non-contiguous inputs via transpose
        query = torch.randn(batch, heads, seq, head_dim, dtype=dtype, device=device).transpose(1, 2)
        key = torch.randn(batch, heads, seq, head_dim, dtype=dtype, device=device).transpose(1, 2)
        value = torch.randn(batch, heads, seq, value_dim, dtype=dtype, device=device).transpose(1, 2)
        g = torch.randn(batch, heads, seq, dtype=dtype, device=device).transpose(1, 2)
        beta = torch.randn(batch, heads, seq, dtype=dtype, device=device).transpose(1, 2)

        assert not query.is_contiguous()
        assert not key.is_contiguous()

        manager = OpManager()

        # Should handle non-contiguous inputs gracefully
        out, _ = manager.call(
            "gated_delta_net_forward",
            query=query, key=key, value=value, g=g, beta=beta,
        )

        assert out.shape == (batch, seq, heads, value_dim)
        assert not torch.isnan(out).any()

    def test_gdn_with_initial_state(self):
        """Test GDN with initial recurrent state."""
        device = torch.device("npu:0")
        dtype = torch.float32
        batch, seq, heads, head_dim, value_dim = 2, 128, 8, 64, 64

        query = torch.randn(batch, seq, heads, head_dim, dtype=dtype, device=device)
        key = torch.randn(batch, seq, heads, head_dim, dtype=dtype, device=device)
        value = torch.randn(batch, seq, heads, value_dim, dtype=dtype, device=device)
        g = torch.randn(batch, seq, heads, dtype=dtype, device=device)
        beta = torch.randn(batch, seq, heads, dtype=dtype, device=device)

        # Create initial state
        initial_state = torch.randn(batch, heads, head_dim, value_dim, dtype=dtype, device=device)

        manager = OpManager()

        out, final_state = manager.call(
            "gated_delta_net_forward",
            query=query, key=key, value=value, g=g, beta=beta,
            initial_state=initial_state,
            output_final_state=True,
        )

        assert out.shape == (batch, seq, heads, value_dim)
        assert final_state is not None
        assert final_state.shape == (batch, heads, head_dim, value_dim)

    def test_gdn_edge_case_empty_sequence(self):
        """Test edge case with very short sequences."""
        device = torch.device("npu:0")
        dtype = torch.float32
        batch, seq, heads, head_dim, value_dim = 1, 1, 4, 32, 32

        query = torch.randn(batch, seq, heads, head_dim, dtype=dtype, device=device)
        key = torch.randn(batch, seq, heads, head_dim, dtype=dtype, device=device)
        value = torch.randn(batch, seq, heads, value_dim, dtype=dtype, device=device)
        g = torch.randn(batch, seq, heads, dtype=dtype, device=device)
        beta = torch.randn(batch, seq, heads, dtype=dtype, device=device)

        manager = OpManager()

        out, _ = manager.call(
            "gated_delta_net_forward",
            query=query, key=key, value=value, g=g, beta=beta,
        )

        assert out.shape == (batch, seq, heads, value_dim)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
