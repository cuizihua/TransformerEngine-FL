# Copyright (c) 2025, BAAI. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Gated Delta Net implementation backed by local GDN implementation."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def is_gated_delta_net_available() -> bool:
    """Return whether the complete FLA-NPU GDN runtime is available."""
    try:
        # Check if fla_npu operators are available
        import fla_npu.ops.ascendc
        import fla_npu.ops.triton

        # Check required operators
        from .gdn_operators import validate_runtime

        return torch.npu.is_available()
    except (ImportError, RuntimeError) as e:
        logger.warning(f"[GDN] Runtime validation failed: {e}")
        return False


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
    """Run GDN while preserving the TE-FL BSHD contract.

    TE-FL receives ``query``, ``key`` and ``value`` in BSHD layout. The GDN
    implementation expects BHSD layout and returns output in BSHD layout.
    ``g`` and ``beta`` remain BSH tensors.

    This vendor implementation intentionally fails closed. Backend selection
    and any reference fallback remain the responsibility of the TE-FL manager
    and its caller.
    """
    # Input validation
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

    # Import local implementation
    from .gdn_impl import flash_gated_delta_rule

    # Log execution details
    B, S, H, D = query.shape

    # Compute scale
    if chunk_size != 2 ** (chunk_size.bit_length() - 1):
        raise ValueError(f"chunk_size must be a power of 2, got {chunk_size}.")

    # Call implementation with layout conversion BSHD -> BHSD
    output, final_state = flash_gated_delta_rule(
        q=query.transpose(1, 2).contiguous(),
        k=key.transpose(1, 2).contiguous(),
        v=value.transpose(1, 2).contiguous(),
        g=g.contiguous(),
        beta=beta.contiguous(),
        scale=None,  # Will be computed inside
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm,
        chunk_size=chunk_size,
    )


    return output.contiguous(), final_state
