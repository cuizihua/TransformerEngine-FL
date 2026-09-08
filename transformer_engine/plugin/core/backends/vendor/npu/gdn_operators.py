# Copyright (c) 2025, BAAI. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""AscendC operator wrappers and runtime validation for GDN."""

from __future__ import annotations

# API version for compatibility checking
GDN_API_VERSION = 1
GDN_PROVIDER = "te_npu_vendor"

# Required AscendC operators for complete GDN implementation
_REQUIRED_ASCENDC_OPS = (
    "chunk_fwd_o",
    "chunk_gated_delta_rule_fwd_h",
    "chunk_gated_delta_rule_bwd_dhu",
    "chunk_bwd_dqkwg",
    "chunk_bwd_dv_local",
    "prepare_wy_repr_bwd_da",
    "prepare_wy_repr_bwd_full",
    "recompute_w_u_fwd",
    "solve_tri",
)

_runtime_info: dict = None


def validate_runtime() -> dict:
    """Validate that all required operators are loaded.

    Returns:
        dict: Runtime information including API version and provider

    Raises:
        RuntimeError: If any required operators are missing
    """
    global _runtime_info

    if _runtime_info is not None:
        return _runtime_info

    try:
        import fla_npu.ops.ascendc as ascendc
    except ImportError as e:
        raise RuntimeError(f"fla_npu.ops.ascendc not available: {e}") from e

    missing = []
    for op_name in _REQUIRED_ASCENDC_OPS:
        if not hasattr(ascendc, op_name):
            missing.append(op_name)

    if missing:
        raise RuntimeError(
            f"GDN runtime incomplete, missing operators in fla_npu.ops.ascendc: {', '.join(missing)}\n"
            f"Please ensure fla_npu is properly installed with all GDN operators."
        )

    _runtime_info = {
        "api_version": GDN_API_VERSION,
        "provider": GDN_PROVIDER,
        "operators": list(_REQUIRED_ASCENDC_OPS),
    }
    return _runtime_info
