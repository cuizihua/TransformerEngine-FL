# Copyright (c) 2025, BAAI. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Gated Delta Net implementation backed by the public FLA-NPU API."""

from __future__ import annotations

from typing import Optional, Tuple

import torch


_MIN_GDN_API_VERSION = 1


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

    from fla_npu.gdn import GDN_API_VERSION, flash_gated_delta_rule, validate_runtime

    if GDN_API_VERSION < _MIN_GDN_API_VERSION:
        raise RuntimeError(
            f"FLA-NPU GDN API version {GDN_API_VERSION} is older than "
            f"the required version {_MIN_GDN_API_VERSION}."
        )
    validate_runtime()

    output, final_state = flash_gated_delta_rule(
        q=query.transpose(1, 2).contiguous(),
        k=key.transpose(1, 2).contiguous(),
        v=value.transpose(1, 2).contiguous(),
        g=g.contiguous(),
        beta=beta.contiguous(),
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm,
        chunk_size=chunk_size,
    )
    return output.contiguous(), final_state
