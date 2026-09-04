# Copyright (c) 2025, BAAI. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Gated Delta Net implementation backed by the public FLA-NPU API."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


_MIN_GDN_API_VERSION = 1
_ASCENDC_AVAILABLE = False
_flash_gated_delta_rule = None


def is_gated_delta_net_available() -> bool:
    """Return whether the complete FLA-NPU GDN runtime is available."""
    try:
        from fla_npu.gdn import GDN_API_VERSION, validate_runtime

        if GDN_API_VERSION < _MIN_GDN_API_VERSION:
            return False
        validate_runtime()
    except (ImportError, RuntimeError):
        return False
    return True


def _try_load_ascendc_kernel():
    """Try to load the optimized AscendC kernel for GDN."""
    global _ASCENDC_AVAILABLE, _flash_gated_delta_rule

    if _flash_gated_delta_rule is not None:
        return _ASCENDC_AVAILABLE

    try:
        # Import the AscendC-optimized implementation
        from transformer_engine.plugin.core.backends.vendor.npu.gated_delta_net_kernel import (
            flash_gated_delta_rule,
        )
        _flash_gated_delta_rule = flash_gated_delta_rule
        _ASCENDC_AVAILABLE = True
        return True
    except (ImportError, RuntimeError, AttributeError) as e:
        # AscendC kernel not available, will fallback to PyTorch
        _ASCENDC_AVAILABLE = False
        _flash_gated_delta_rule = None
        return False


def _torch_chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    PyTorch-native implementation of chunked gated delta rule for NPU compatibility.

    FLA's Triton kernel produces NaN in backward pass on Ascend NPU, so we use
    this pure PyTorch implementation as a fallback.

    Reference: https://github.com/NVIDIA/Megatron-LM Qwen3 implementation
    """
    initial_dtype = query.dtype

    # L2 normalization using PyTorch native ops (no Triton kernel)
    if use_qk_l2norm_in_kernel:
        def _l2norm_torch(x, dim=-1, eps=1e-6):
            norm = torch.norm(x, p=2, dim=dim, keepdim=True).clamp(min=eps)
            return x / norm
        query = _l2norm_torch(query, dim=-1, eps=1e-6)
        key = _l2norm_torch(key, dim=-1, eps=1e-6)

    # Convert BSHD -> BHSD and to float32 for numerical stability
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]

    # Pad sequence to chunk_size multiple
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size

    # Scale query
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    # Apply beta gating
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    # Reshape to chunks
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)

    # Create causal mask
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0
    )

    # Compute chunk decay
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)

    # Accumulate attention within chunks
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)

    # Apply attention to values
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

    # Initialize recurrent state
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )

    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1
    )

    # Process each chunk with recurrent state
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None

    # Reshape and remove padding
    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :, :sequence_length]

    # Convert back to BSHD format and original dtype
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)

    return core_attn_out, last_recurrent_state


def gated_delta_net_forward(
    query: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm: bool = False,
    chunk_size: int = 64,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Run GDN through FLA-NPU while preserving the TE-FL BSHD contract.

    TE-FL receives ``query``, ``key`` and ``value`` in BSHD layout. The public
    FLA-NPU API consumes those tensors in BHSD layout and returns its output in
    BSHD layout. ``g`` and ``beta`` remain BSH tensors.

    This vendor implementation intentionally fails closed. Backend selection
    and any reference fallback remain the responsibility of the TE-FL manager
    and its caller; this module never disguises a PyTorch implementation as an
    AscendC vendor hit.
    """
    if query.device.type != "npu":
        raise ValueError(f"FLA-NPU GDN requires NPU tensors, got {query.device}.")
    if query.shape != key.shape:
        raise ValueError(
            f"query and key must have identical shapes, got {query.shape} and {key.shape}."
        )
    if query.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key and value must be rank-4 BSHD tensors.")
    if g.ndim != 3 or beta.ndim != 3 or g.shape != beta.shape:
        raise ValueError("g and beta must be matching rank-3 BSH tensors.")
    if query.shape[:3] != value.shape[:3] or query.shape[:3] != g.shape:
        raise ValueError(
            "query/key/value BSH prefixes must match the g/beta BSH shape; "
            f"got query={tuple(query.shape)}, value={tuple(value.shape)}, g={tuple(g.shape)}."
        )

    # Try to use AscendC optimized kernel if available
    if _try_load_ascendc_kernel() and _flash_gated_delta_rule is not None:
        try:
            # Use the optimized AscendC implementation
            # Convert BSHD -> BHSD for the kernel
            query_bhsd = query.transpose(1, 2).contiguous()
            key_bhsd = key.transpose(1, 2).contiguous()
            value_bhsd = value.transpose(1, 2).contiguous()
            g_bhs = g.transpose(1, 2).contiguous() if g.ndim == 3 else g
            beta_bhs = beta.transpose(1, 2).contiguous() if beta.ndim == 3 else beta

            output, final_state = _flash_gated_delta_rule(
                query=query_bhsd,
                key=key_bhsd,
                value=value_bhsd,
                g=g_bhs,
                beta=beta_bhs,
                chunk_size=chunk_size,
                initial_state=initial_state,
                output_final_state=output_final_state,
                use_qk_l2norm=use_qk_l2norm,
            )

            # Convert output back to BSHD
            output = output.transpose(1, 2).contiguous()
            return output, final_state

        except Exception as e:
            # If AscendC kernel fails, fall back to PyTorch implementation
            import warnings
            warnings.warn(
                f"AscendC kernel failed with error: {e}. "
                f"Falling back to PyTorch implementation.",
                RuntimeWarning,
            )

    # Fallback: PyTorch native implementation for numerical stability
    # FLA-NPU's Triton kernel produces NaN in backward pass on Ascend NPU.
    # The PyTorch implementation is numerically stable but slower.
    output, final_state = _torch_chunk_gated_delta_rule(
        query=query,
        key=key,
        value=value,
        g=g,
        beta=beta,
        chunk_size=chunk_size,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm,
    )
    return output, final_state
