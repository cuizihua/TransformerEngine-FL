# Copyright (c) 2026, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""Integration tests for the TE-FL to FLA-NPU GDN vendor backend."""

import pytest
import torch


try:
    import torch_npu  # noqa: F401

    NPU_AVAILABLE = torch.npu.is_available() and torch.npu.device_count() > 0
except (ImportError, AttributeError):
    NPU_AVAILABLE = False


@pytest.mark.skipif(not NPU_AVAILABLE, reason="NPU not available")
def test_fla_npu_gdn_forward_backward_dispatch():
    """The manager must select the real provider and preserve all input grads."""
    from fla_npu.gdn import GDN_PROVIDER, validate_runtime
    from transformer_engine.plugin.core.manager import OpManager

    runtime = validate_runtime()
    assert GDN_PROVIDER == "fla_npu"
    assert runtime["provider"] == "fla_npu"

    torch.manual_seed(42)
    device = torch.device("npu:0")
    batch, seq, heads, key_dim, value_dim = 1, 128, 4, 128, 128
    tensors = [
        torch.randn(batch, seq, heads, key_dim, device=device, dtype=torch.bfloat16),
        torch.randn(batch, seq, heads, key_dim, device=device, dtype=torch.bfloat16),
        torch.randn(batch, seq, heads, value_dim, device=device, dtype=torch.bfloat16),
        torch.randn(batch, seq, heads, device=device, dtype=torch.float32) * 0.05 - 0.1,
        torch.sigmoid(torch.randn(batch, seq, heads, device=device, dtype=torch.float32)),
    ]
    for tensor in tensors:
        tensor.requires_grad_(True)
    query, key, value, g, beta = tensors

    manager = OpManager()
    candidates = manager.resolve_candidates("gated_delta_net_forward")
    assert candidates[0].impl_id == "vendor.npu.gdn_fwd"

    output, final_state = manager.call(
        "gated_delta_net_forward",
        query=query,
        key=key,
        value=value,
        g=g,
        beta=beta,
        output_final_state=False,
        use_qk_l2norm=False,
        chunk_size=64,
    )
    assert manager._get_last_impl_id("gated_delta_net_forward") == "vendor.npu.gdn_fwd"
    assert output.shape == value.shape
    assert output.dtype == value.dtype
    assert output.device == value.device
    assert output.is_contiguous()
    assert final_state is None

    grad_output = torch.randn_like(output) * 0.05
    output.backward(grad_output)
    for tensor in tensors:
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
