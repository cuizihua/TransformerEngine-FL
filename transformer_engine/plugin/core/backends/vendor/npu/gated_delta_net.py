# Copyright (c) 2025, BAAI. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Gated Delta Net implementation using installed FLA-NPU AscendC kernels."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def is_gated_delta_net_available() -> bool:
    """Return whether FLA-NPU GDN runtime is available."""
    try:
        import fla_npu.ops.ascendc
        # Check if the required GDN kernels exist
        return hasattr(fla_npu.ops.ascendc, 'chunk_gated_delta_rule_fwd_h')
    except ImportError:
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
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm: bool = False,
    chunk_size: int = 64,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Run GDN using installed FLA-NPU AscendC kernels.

    Args:
        query: [B, S, H, D] Query tensor in BSHD layout
        key: [B, S, H, D] Key tensor in BSHD layout
        value: [B, S, H, D] Value tensor in BSHD layout
        g: [B, S, H] Gating tensor
        beta: [B, S, H] Beta tensor
        initial_state: Optional initial recurrent state
        use_qk_l2norm: Whether to apply L2 normalization to Q and K
        chunk_size: Chunk size for processing

    Returns:
        output: [B, S, H, D] Output tensor in BSHD layout
        final_state: Optional final recurrent state
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

    # Try to use FLA-NPU AscendC fused kernel (aclnnChunkGatedDeltaRuleFwd)
    try:
        import fla_npu.ops.ascendc

        if hasattr(fla_npu.ops.ascendc, 'npu_chunk_gated_delta_rule_fwd'):
            # Use the fused AscendC kernel which implements all 6 steps in one kernel
            # This is the Phase 6 fusion kernel that replaces the entire operator chain

            # IMPORTANT: The fused kernel expects BHSD format, but we receive BSHD
            # Need to transpose: [B, S, H, D] -> [B, H, S, D]
            query_bhsd = query.transpose(1, 2).contiguous()  # [B, S, H, K] -> [B, H, S, K]
            key_bhsd = key.transpose(1, 2).contiguous()      # [B, S, H, K] -> [B, H, S, K]
            value_bhsd = value.transpose(1, 2).contiguous()  # [B, S, H, V] -> [B, H, S, V]

            # g and beta need to be transposed from [B, S, H] -> [B, H, S]
            # But the fused kernel expects g: [B, T, H] and beta: [B, T, H] in BTH format!
            # Let me check the actual requirement...
            # From the code analysis: g: [B, T, Hv], beta: [B, T, Hv]
            # So we DON'T transpose g and beta - keep them as [B, S, H]
            g_bth = g  # Keep as [B, S, H] = [B, T, H]
            beta_bth = beta  # Keep as [B, S, H] = [B, T, H]

            # Calculate scale if not provided
            scale = 1.0 / (query.shape[-1] ** 0.5)

            # Call the fused kernel
            # Returns: (o, final_state, g_cumsum, A)
            # o: [B, Hv, T, V] output in BHTV format
            o, final_state_out, g_cumsum, A = fla_npu.ops.ascendc.npu_chunk_gated_delta_rule_fwd(
                q=query_bhsd,       # [B, H, T, K]
                k=key_bhsd,         # [B, H, T, K]
                v=value_bhsd,       # [B, H, T, V]
                g=g_bth,            # [B, T, H]
                beta=beta_bth,      # [B, T, H]
                initial_state=initial_state,
                output_final_state=output_final_state,
                chunk_size=chunk_size,
                cu_seqlens=None,    # Not using varlen mode
                chunk_indices=None,
                scale=scale,
            )

            # Convert output from [B, H, T, V] back to [B, T, H, V] (BSHD format)
            output = o.transpose(1, 2).contiguous()

            # print("=== FLA-NPU AscendC 融合算子使用成功 ===", flush=True)
            return output, final_state_out

    except (ImportError, AttributeError, RuntimeError) as e:
        # Fall back to PyTorch implementation if AscendC kernel fails
        import logging
        logging.warning(f"FLA-NPU AscendC 融合算子失败，回退到 PyTorch: {e}")

    # PyTorch fallback implementation
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
