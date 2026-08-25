# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""DeepSeek-V4 MLA sparse prefill, served from a prebuilt code object.

Same four kernels and the same validation as the JIT ops in
``aiter/ops/pa_sparse_prefill_opus.py``, but the device code is compiled ahead
of time into ``hsa/gfx950/dsv4_mla_prefill/dsv4_mla_prefill.co``.  The
first call therefore costs no compile, and -- the reason this exists -- the
attention kernel's schedule is fixed at ship time.  It sits at 504 of 512
ArchVGPRs with zero scratch, so a toolchain that costs it eight registers spills
the inner loop with no diagnostic.

The attention op is ``dsv4_mla_prefill``; the JIT module spells the same kernel
``pa_sparse_prefill_fp8_h40_opus``.  The three helpers keep their JIT names --
``pa_fp8_q_pack`` -- because
they are not prefill and a caller has no reason to care which build produced
them.  That does mean the two modules must not both be imported:
``aiter/__init__`` would bind whichever ran last for those three.  Pick one.

Requires gfx950 and ``H >= PA_FP8_MIN_H``; the block is 128 heads wide and narrower shapes
are rejected rather than run slowly.  See the JIT module for the full argument
contract, which is unchanged.
"""

import torch

from ..jit.core import compile_ops
from ..jit.utils.chip_info import get_gfx_runtime
from ..jit.utils.torch_guard import torch_compile_guard

_MODULE = "module_dsv4_mla_prefill"

# Part of the op's contract rather than of this build, so these keep the names
# the JIT module exports -- vLLM reads both to decide admission. Keep MIN_H in
# sync with PA_FP8_H40_MIN_H in the kernel header, which is where the
# admission threshold is actually compiled in.
PA_FP8_MIN_H = 16
PA_FP8_GLOBAL64 = True


@compile_ops(_MODULE, fc_name="dsv4_mla_prefill_warmup", ffi_type="ctypes")
def _warmup() -> int: ...


@compile_ops(_MODULE, fc_name="dsv4_mla_q_pack_fwd", ffi_type="ctypes")
def _q_pack(q_nope_bf16: torch.Tensor, out: torch.Tensor) -> int: ...


@compile_ops(_MODULE, fc_name="dsv4_mla_prefill_fwd", ffi_type="ctypes")
def _h40_prefill(
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
) -> int: ...


def _empty_i32(like: torch.Tensor) -> torch.Tensor:
    """Sentinel for an omitted optional tensor argument (numel 0 == not given)."""
    return torch.empty(0, dtype=torch.int32, device=like.device)


def _require_gfx950(op: str) -> None:
    gfx = get_gfx_runtime()
    if gfx != "gfx950":
        raise RuntimeError(f"{op} requires gfx950, got {gfx}")


def _q_pack_fake(q_nope_bf16: torch.Tensor, out: torch.Tensor) -> None:
    return None


@torch_compile_guard(mutates_args=["out"], gen_fake=_q_pack_fake)
def pa_fp8_q_pack(q_nope_bf16: torch.Tensor, out: torch.Tensor) -> None:
    """Pack ``[..., 448]`` bf16 Q into the ``[..., 512]`` fp8 + E8M0 layout.

    The prefill kernel packs Q in its own prologue, so this is only for callers
    that want the packed buffer for something else.
    """
    _require_gfx950("pa_fp8_q_pack")
    _q_pack(q_nope_bf16, out)


def _h40_fake(
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
) -> torch.Tensor:
    if out is not None:
        return out
    n, h = q_nope.shape[0], q_nope.shape[1]
    return torch.empty(n, h, 512, dtype=torch.bfloat16, device=q_nope.device)


@torch_compile_guard(mutates_args=["out"], gen_fake=_h40_fake)
def dsv4_mla_prefill(
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
) -> torch.Tensor:
    """DeepSeek-V4 h40 sparse prefill attention, both GEMMs on fp8.

    The KV cache is read exactly as it was written: the kernel requantises each
    staged tile in LDS and takes its softmax frame from that tile, so no pass
    has to flatten the seven block-64 scales a page stores per token.
    ``kv_max_e`` is kept for call compatibility and is ignored; pass a zeroed
    int32 device scalar.
    """
    _require_gfx950("dsv4_mla_prefill")
    if out is None:
        out = torch.empty(
            q_nope.shape[0],
            q_nope.shape[1],
            512,
            dtype=torch.bfloat16,
            device=q_nope.device,
        )
    _h40_prefill(
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
    )
    return out


# Registering a code object is illegal while a stream is capturing, and with
# cudagraphs the first call can land inside one. Force it here, at import.
_warmup()
