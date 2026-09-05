"""AscendC training kernels for Qwen3.5 gated delta net.

The FlagOS FLA source tree contains the model-facing autograd wrappers and
Triton preprocessing kernels.  The decoupled ``fla_npu`` runtime is loaded
from the parity workspace dependency so it does not go through the
torch/torch_npu dispatcher ABI used by the older installed wheel.

Both roots can be overridden for packaged deployments.  Keeping this setup in
one module also lets the generic PyTorch implementation remain the import-time
fallback when the optional NPU dependency is unavailable.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path
from typing import Optional, Tuple

# Force set FLA-NPU environment variables before any other imports
os.environ.setdefault('FLAGOS_FLA_NPU_SOURCE_ROOT', '/data/tiankuan/tk-czh/flash-linear-attention-npu')
os.environ.setdefault('FLAGOS_FLA_NPU_DIRECT_ROOT', '/data/tiankuan/tk-czh/flash-linear-attention-npu/torch_custom/fla_npu/build/lib.linux-aarch64-cpython-312')

import torch


_WORKSPACE_ROOT = Path(__file__).resolve().parents[7]
_FLA_SOURCE_ROOT = Path(
    os.environ.get(
        "FLAGOS_FLA_NPU_SOURCE_ROOT",
        _WORKSPACE_ROOT
        / "fsos_base_line"
        / "src"
        / "flash-linear-attention-npu-c2e3d83f",
    )
).resolve()
_FLA_DIRECT_ROOT = Path(
    os.environ.get(
        "FLAGOS_FLA_NPU_DIRECT_ROOT",
        _WORKSPACE_ROOT / "mindspeed" / "flash-linear-attention-npu" / "build" / "lib",
    )
).resolve()


def _prepend_import_root(path: Path) -> None:
    if not path.is_dir():
        raise ImportError(f"Required FLA-NPU import root does not exist: {path}")
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)


def _load_wrappers():
    import sys
    print("[DEBUG] _load_wrappers() called", file=sys.stderr, flush=True)
    print(f"[DEBUG] _FLA_DIRECT_ROOT: {_FLA_DIRECT_ROOT}", file=sys.stderr, flush=True)
    print(f"[DEBUG] _FLA_SOURCE_ROOT: {_FLA_SOURCE_ROOT}", file=sys.stderr, flush=True)
    print(f"[DEBUG] _FLA_DIRECT_ROOT exists: {_FLA_DIRECT_ROOT.exists()}", file=sys.stderr, flush=True)
    print(f"[DEBUG] _FLA_SOURCE_ROOT exists: {_FLA_SOURCE_ROOT.exists()}", file=sys.stderr, flush=True)

    # Insert the direct runtime first, then the FlagOS source root.  The latter
    # intentionally wins for ``fla.ops.triton`` while only the direct root
    # provides a top-level ``fla_npu`` package.
    _prepend_import_root(_FLA_DIRECT_ROOT)
    _prepend_import_root(_FLA_SOURCE_ROOT)

    print(f"[DEBUG] sys.path after prepend: {sys.path[:5]}", file=sys.stderr, flush=True)

    loaded_fla_npu = sys.modules.get("fla_npu")
    print(f"[DEBUG] loaded_fla_npu: {loaded_fla_npu}", file=sys.stderr, flush=True)
    if loaded_fla_npu is not None:
        loaded_path = Path(getattr(loaded_fla_npu, "__file__", "")).resolve()
        print(f"[DEBUG] fla_npu loaded from: {loaded_path}", file=sys.stderr, flush=True)
        if _FLA_DIRECT_ROOT not in loaded_path.parents:
            raise ImportError(
                "An incompatible fla_npu package was imported before the direct runtime: "
                f"{loaded_path}"
            )

    print(f"[DEBUG] About to import fla.ops.triton modules", file=sys.stderr, flush=True)
    from fla.ops.triton.triton_core.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
    from fla.ops.triton.triton_core.cumsum import chunk_local_cumsum
    from fla.ops.triton.triton_core.l2norm import l2norm_bwd, l2norm_fwd
    from fla.ops.triton.triton_core.solve_tril_fast import solve_tril_npu
    from fla.ops.triton.triton_core.utils import (
        autocast_custom_bwd,
        autocast_custom_fwd,
        input_guard,
    )

    triton_shim = types.ModuleType("fla_npu.ops.triton")
    triton_exports = {
        "autocast_custom_bwd": autocast_custom_bwd,
        "autocast_custom_fwd": autocast_custom_fwd,
        "chunk_local_cumsum": chunk_local_cumsum,
        "chunk_scaled_dot_kkt_fwd": chunk_scaled_dot_kkt_fwd,
        "input_guard": input_guard,
        "l2norm_bwd": l2norm_bwd,
        "l2norm_fwd": l2norm_fwd,
        "solve_tril_npu": solve_tril_npu,
    }
    for name, value in triton_exports.items():
        setattr(triton_shim, name, value)
    sys.modules[triton_shim.__name__] = triton_shim

    print(f"[DEBUG] About to import fla_npu.ops.ascendc", file=sys.stderr, flush=True)
    try:
        import fla_npu.ops.ascendc as ascendc
        print(f"[DEBUG] Successfully imported fla_npu.ops.ascendc from: {ascendc.__file__}", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[DEBUG] Failed to import fla_npu.ops.ascendc: {e}", file=sys.stderr, flush=True)
        raise

    original_fwd_h = ascendc.chunk_gated_delta_rule_fwd_h
    if not getattr(original_fwd_h, "_flagos_abi_compat", False):

        def compatible_fwd_h(*args, **kwargs):
            # The FlagOS wrapper exposes these options, while the decoupled
            # runtime has the production values fixed in its current ABI.
            kwargs.pop("save_new_value", None)
            kwargs.pop("use_exp2", None)
            kwargs.pop("transpose_state_layout", None)
            return original_fwd_h(*args, **kwargs)

        compatible_fwd_h._flagos_abi_compat = True
        ascendc.chunk_gated_delta_rule_fwd_h = compatible_fwd_h
        ascendc.npu_chunk_gated_delta_rule_fwd_h = compatible_fwd_h

    wrapper_path = _FLA_SOURCE_ROOT / "examples" / "flash_gated_delta_rule.py"
    print(f"[DEBUG] Looking for wrapper at: {wrapper_path}", file=sys.stderr, flush=True)
    print(f"[DEBUG] Wrapper file exists: {wrapper_path.exists()}", file=sys.stderr, flush=True)
    spec = importlib.util.spec_from_file_location("_flagos_fla_npu_gdn_wrappers", wrapper_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load FLA-NPU wrappers from {wrapper_path}")
    wrappers = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = wrappers
    spec.loader.exec_module(wrappers)
    print(f"[DEBUG] Successfully loaded wrappers from: {wrapper_path}", file=sys.stderr, flush=True)

    def recompute_w_u_no_sync(
        k,
        v,
        beta,
        A,
        g,
        *,
        chunk_size,
        cu_seqlens,
        chunk_indices,
    ):
        # The example wrapper synchronizes for standalone diagnostics.  A
        # training path must preserve stream asynchrony.
        return wrappers.ascendc_recompute_w_u_fwd(
            k,
            v,
            beta,
            A,
            chunk_size,
            g=g,
            gk=None,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )

    wrappers.recompute_w_u = recompute_w_u_no_sync
    return wrappers


# Try to load AscendC kernels
import sys
print("[DEBUG] Starting to load AscendC kernels", file=sys.stderr, flush=True)
try:
    _WRAPPERS = _load_wrappers()
    print("[DEBUG] _load_wrappers() succeeded", file=sys.stderr, flush=True)
    _flash_gated_delta_rule_impl = _WRAPPERS.flash_gated_delta_rule
    print("[DEBUG] Got flash_gated_delta_rule implementation", file=sys.stderr, flush=True)
    ASCENDC_AVAILABLE = True
    print("[DEBUG] ASCENDC_AVAILABLE = True", file=sys.stderr, flush=True)
except (ImportError, RuntimeError, AttributeError) as e:
    _WRAPPERS = None
    _flash_gated_delta_rule_impl = None
    ASCENDC_AVAILABLE = False
    print(
        f"[ERROR] AscendC kernel for GatedDeltaNet not available: {e}. "
        "Will use PyTorch fallback implementation.",
        file=sys.stderr, flush=True
    )
    import traceback
    print(f"[DEBUG] Full traceback:\n{traceback.format_exc()}", file=sys.stderr, flush=True)


def flash_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Run GDN with AscendC optimized kernel.

    Args:
        query: [B, H, S, D_k] Query tensor in BHSD format
        key: [B, H, S, D_k] Key tensor in BHSD format
        value: [B, H, S, D_v] Value tensor in BHSD format
        g: [B, H, S] or [B, S, H] Gating tensor
        beta: [B, H, S] or [B, S, H] Beta tensor
        chunk_size: Chunk size for processing
        initial_state: Optional initial recurrent state
        output_final_state: Whether to output final state
        use_qk_l2norm: Whether to apply L2 normalization to Q and K

    Returns:
        output: [B, H, S, D_v] Output tensor in BHSD format
        final_state: Optional final recurrent state
    """
    if not ASCENDC_AVAILABLE or _flash_gated_delta_rule_impl is None:
        raise RuntimeError(
            "AscendC kernel is not available. This should not be called directly. "
            "Use gated_delta_net_forward which has PyTorch fallback."
        )

    return _flash_gated_delta_rule_impl(
        query=query,
        key=key,
        value=value,
        g=g,
        beta=beta,
        chunk_size=chunk_size,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm=use_qk_l2norm,
    )


__all__ = ["flash_gated_delta_rule", "ASCENDC_AVAILABLE"]
