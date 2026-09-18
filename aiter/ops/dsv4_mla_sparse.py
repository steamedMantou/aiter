# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""DeepSeek-V4 sparse MLA attention, served from a prebuilt code object.

Serves both phases.  A decode step is the same operation with a much smaller N
-- N query rows, each gathering its own index list plus its own sliding window
out of the same pool -- so all 64 of its attention layers go through this op
too, and there is no second kernel for them.  The op was called
``dsv4_mla_prefill`` until that stopped being true of half its traffic.

The device code is compiled ahead of time under
``hsa/gfx950/dsv4_mla_sparse/``.  The first call therefore costs no device
compile, and the attention schedule is fixed at ship time.  It uses the full
register budget with zero scratch, and its ISA is checked to contain no
``v_accvgpr_read/write/mov`` instructions.

Requires gfx950 and ``H >= PA_FP8_MIN_H``; the block is 128 heads wide and narrower shapes
are rejected rather than run slowly.  See the JIT module for the full argument
contract, which is unchanged.
"""

import torch

from ..jit.core import compile_ops
from ..jit.utils.chip_info import get_gfx_runtime
from ..jit.utils.torch_guard import torch_compile_guard

_MODULE = "module_dsv4_mla_sparse"

# Part of the op's contract rather than of this build, so these keep the names
# the JIT module exports -- vLLM reads both to decide admission. Keep MIN_H in
# sync with PA_SPARSE_MLA_MIN_H in the kernel header, which is where the
# admission threshold is actually compiled in.
PA_FP8_MIN_H = 16
PA_FP8_GLOBAL64 = True
# Whether this build takes the optional `row_map` argument.  A caller reads it
# instead of probing the signature: a keyword the op does not have fails at
# *call* time, which under cudagraphs is the worst place to find out.
PA_FP8_HAS_ROW_MAP = True


@compile_ops(_MODULE, fc_name="dsv4_mla_xcc_count", ffi_type="ctypes")
def _xcc_count(device_id: int) -> int: ...


_XCC: dict[int, int] = {}


def xcc_count(device_id: int = 0) -> int:
    """XCCs on the device, in its *current* compute partition.

    A caller placing work by XCC needs this divisor, and both ways of getting it
    without asking here are worse: torch does not surface it, and reaching
    libamdhip64 by ctypes means writing hipDeviceAttributeNumberOfXccs' numeric
    value into Python, where it is a worse constant than the one it replaces --
    it moves with the ROCm version.  Here the compiler resolves the enum against
    the headers this was built with.

    The partition is set out of band (amd-smi), so this describes how the device
    is configured right now, not the chip -- which is the reason a caller should
    not spell it as a literal.  Cached per device: a partition cannot change
    under a live context.

    Raises RuntimeError rather than returning a fallback: a caller that silently
    used the wrong divisor would build a permutation that is still valid (so
    nothing fails) and simply stops placing anything, which is invisible.
    """
    cached = _XCC.get(device_id)
    if cached is None:
        got = int(_xcc_count(int(device_id)))
        if got <= 0:
            raise RuntimeError(
                f"could not read the XCC count for device {device_id} (rc={got})"
            )
        _XCC[device_id] = cached = got
    return cached


@compile_ops(_MODULE, fc_name="dsv4_mla_sparse_fwd", ffi_type="ctypes")
def _sparse_mla(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    unified_kv_nope: torch.Tensor,
    unified_kv_rope: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv_nope: torch.Tensor,
    kv_rope: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor,
    kv_max_e: torch.Tensor,
    out: torch.Tensor,
    softmax_scale: float,
    kv_lens_prefix: torch.Tensor,
    kv_lens_extend: torch.Tensor,
    kv_stride_q_prefix: int,
    kv_stride_q_extend: int,
    page_shift_prefix: int,
    rows_per_page_prefix: int,
    scale_off_prefix: int,
    page_shift_extend: int,
    rows_per_page_extend: int,
    scale_off_extend: int,
    row_map: torch.Tensor,
) -> int: ...


def _empty_i32(like: torch.Tensor) -> torch.Tensor:
    """Sentinel for an omitted optional tensor argument (numel 0 == not given)."""
    return torch.empty(0, dtype=torch.int32, device=like.device)


def _require_gfx950(op: str) -> None:
    gfx = get_gfx_runtime()
    if gfx != "gfx950":
        raise RuntimeError(f"{op} requires gfx950, got {gfx}")


def _sparse_mla_fake(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    unified_kv_nope: torch.Tensor,
    unified_kv_rope: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv_nope: torch.Tensor,
    kv_rope: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor,
    kv_max_e: torch.Tensor,
    softmax_scale: float,
    out: torch.Tensor | None = None,
    kv_lens_prefix: torch.Tensor | None = None,
    kv_lens_extend: torch.Tensor | None = None,
    kv_stride_q_prefix: int = 0,
    kv_stride_q_extend: int = 0,
    page_shift_prefix: int = 0,
    rows_per_page_prefix: int = 1,
    scale_off_prefix: int = 448,
    page_shift_extend: int = 0,
    rows_per_page_extend: int = 1,
    scale_off_extend: int = 448,
    row_map: torch.Tensor | None = None,
) -> torch.Tensor:
    if out is not None:
        return out
    n, h = q_nope.shape[0], q_nope.shape[1]
    return torch.empty(n, h, 512, dtype=torch.bfloat16, device=q_nope.device)


@torch_compile_guard(mutates_args=["out"], gen_fake=_sparse_mla_fake)
def dsv4_mla_sparse(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    unified_kv_nope: torch.Tensor,
    unified_kv_rope: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv_nope: torch.Tensor,
    kv_rope: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor,
    kv_max_e: torch.Tensor,
    softmax_scale: float,
    out: torch.Tensor | None = None,
    kv_lens_prefix: torch.Tensor | None = None,
    kv_lens_extend: torch.Tensor | None = None,
    kv_stride_q_prefix: int = 0,
    kv_stride_q_extend: int = 0,
    page_shift_prefix: int = 0,
    rows_per_page_prefix: int = 1,
    scale_off_prefix: int = 448,
    page_shift_extend: int = 0,
    rows_per_page_extend: int = 1,
    scale_off_extend: int = 448,
    row_map: torch.Tensor | None = None,
) -> torch.Tensor:
    """DeepSeek-V4 sparse MLA attention, both GEMMs on fp8.

    The KV cache is read exactly as it was written: the kernel requantises each
    staged tile in LDS and takes its softmax frame from that tile, so no pass
    has to flatten the seven block-64 scales a page stores per token.
    ``kv_max_e`` is kept for call compatibility and is ignored; pass a zeroed
    int32 device scalar.

    """
    _require_gfx950("dsv4_mla_sparse")
    if out is None:
        out = torch.empty(
            q_nope.shape[0],
            q_nope.shape[1],
            512,
            dtype=torch.bfloat16,
            device=q_nope.device,
        )
    _sparse_mla(
        q_nope,
        q_rope,
        unified_kv_nope,
        unified_kv_rope,
        kv_indices_prefix,
        kv_indptr_prefix,
        kv_nope,
        kv_rope,
        kv_indices_extend,
        kv_indptr_extend,
        attn_sink,
        kv_max_e,
        out,
        softmax_scale,
        _empty_i32(q_nope) if kv_lens_prefix is None else kv_lens_prefix,
        _empty_i32(q_nope) if kv_lens_extend is None else kv_lens_extend,
        kv_stride_q_prefix,
        kv_stride_q_extend,
        page_shift_prefix,
        rows_per_page_prefix,
        scale_off_prefix,
        page_shift_extend,
        rows_per_page_extend,
        scale_off_extend,
        _empty_i32(q_nope) if row_map is None else row_map,
    )
    return out
