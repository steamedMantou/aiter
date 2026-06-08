# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""MXFP4 MQA-logits ops -- the FP4 counterparts of the FP8
`aiter.ops.triton.attention.fp8_mqa_logits` (dense) and
`aiter.ops.triton.attention.pa_mqa_logits.deepgemm_fp8_paged_mqa_logits` (paged).

Both are backed by a hand-written gfx950 (CDNA4) HIP kernel that ships with aiter
(csrc/kernels/fp4_mqa_logits_kernels.cu, built on first use by the aiter jit as
``module_fp4_mqa_logits``), hence they live under ``aiter.ops`` rather than
``aiter.ops.triton``. Q and K are MXFP4 (packed e2m1 data + e8m0 micro-scale per
32 elems), so unlike the FP8 ops there is no separate per-row ``kv_scales``
tensor -- the scales are the ``q_s`` / ``kv_s`` exponent planes.

  fp4_mqa_logits        -- dense / ragged prefill path
  fp4_paged_mqa_logits  -- paged decode path (KV block_size == 1)
  make_fp4_kv_cache     -- pack MXFP4 KV planes into the fused paged cache
"""
import functools

import torch
from torch import Tensor

from ..jit.core import compile_ops

MD_NAME = "module_fp4_mqa_logits"

# Fused paged-cache row stride: align(D/2 + D/32, 16) = align(68, 16). Must match
# FP4_KV_STRIDE in csrc/py_itfs_cu/fp4_mqa_logits.cu.
KV_STRIDE = 80


@functools.lru_cache(maxsize=None)
def _num_cu(device=None):
    return torch.cuda.get_device_properties(device).multi_processor_count


def pick_split_kv(b, n, target_per_cu=8, cap=2048, device=None):
    """Pick split_kv from the launch fan-out only (never the kv/context length).

    The grid is ``b * n * split_kv``; sizing split from ``(b, n, device)`` alone
    keeps the grid fixed per batch size, so one CUDA graph can be captured per
    batch size and replayed for any kv length (the kernel slices the per-request
    tile range at runtime and empty splits early-return).
    """
    target = _num_cu(device) * target_per_cu
    split = max(1, target // (b * n))
    return int(min(split, cap))


# ---------------------------------------------------------------------------
# Raw C++ bindings (compiled on first use). Argument *names* must match the
# pybind py::arg names in csrc/include/rocm_ops.hpp (FP4_MQA_LOGITS_PYBIND).
# ---------------------------------------------------------------------------
@compile_ops("module_fp4_mqa_logits", fc_name="fp4_mqa_logits")
def _fp4_mqa_logits(
    q_p: Tensor,
    q_s: Tensor,
    kv_p: Tensor,
    kv_s: Tensor,
    weights: Tensor,
    cu_starts: Tensor,
    cu_ends: Tensor,
    out: Tensor,
    split_kv: int,
    core: int,
) -> Tensor: ...


@compile_ops("module_fp4_mqa_logits", fc_name="fp4_paged_mqa_logits")
def _fp4_paged_mqa_logits(
    q_p: Tensor,
    q_s: Tensor,
    kv_cache: Tensor,
    block_tables: Tensor,
    weights: Tensor,
    context_lens: Tensor,
    out_logits: Tensor,
    max_model_len: int,
    split_kv: int,
) -> Tensor: ...


def fp4_mqa_logits(
    q_p,
    q_s,
    kv_p,
    kv_s,
    weights,
    cu_starts,
    cu_ends,
    clean_logits=True,
    split_kv=None,
    core="32",
):
    """Dense (non-paged) MXFP4 MQA-logits.

    Computes logits[m, n] = sum_h relu(Q[m,h,:] . K[n,:]) * weights[m,h]
    for cu_starts[m] <= n < cu_ends[m], else -inf.

    q_p:        [seq_len, NUM_HEADS, HEAD_SIZE // 2],  uint8  (MXFP4 e2m1 data)
    q_s:        [seq_len, NUM_HEADS, HEAD_SIZE // 32], uint8  (MXFP4 e8m0 scale)
    kv_p:       [seq_len_kv, HEAD_SIZE // 2],  uint8
    kv_s:       [seq_len_kv, HEAD_SIZE // 32], uint8
    weights:    [seq_len, NUM_HEADS], float32
    cu_starts:  [seq_len], int32, per-row valid start (inclusive)
    cu_ends:    [seq_len], int32, per-row valid end (exclusive)
    clean_logits: if True, positions outside [cu_starts[i], cu_ends[i]) in row i
                  are written as -inf. If False they are left undefined and the
                  caller must mask/ignore them.
    split_kv:   optional kv-split factor (None -> graph-stable heuristic).
    core:       "32" (32x32x64 matrix core, default & fastest) or "16"
                (16x16x128 core).

    Returns logits [seq_len, seq_len_kv] float32.
    """
    seq_len, num_heads = weights.shape
    seq_len_kv = kv_p.shape[0]
    if clean_logits:
        out = torch.full(
            (seq_len, seq_len_kv),
            float("-inf"),
            device=q_p.device,
            dtype=torch.float32,
        )
    else:
        out = torch.empty(
            (seq_len, seq_len_kv), device=q_p.device, dtype=torch.float32
        )
    if split_kv is None:
        split_kv = pick_split_kv(seq_len, 1)
    core_i = 16 if str(core) == "16" else 32
    return _fp4_mqa_logits(
        q_p,
        q_s,
        kv_p,
        kv_s,
        weights,
        cu_starts,
        cu_ends,
        out,
        int(split_kv),
        int(core_i),
    )


def make_fp4_kv_cache(kv_p, kv_s):
    """Pack MXFP4 KV planes into the 16B-aligned fused cache the paged kernel reads.

    kv_p: [num_blocks, HEAD_SIZE // 2],  uint8 (e2m1 data)
    kv_s: [num_blocks, HEAD_SIZE // 32], uint8 (e8m0 scale)
    returns: [num_blocks, KV_STRIDE] uint8 (data | scale | pad), contiguous.
    """
    num_blocks, dp = kv_p.shape
    ds = kv_s.shape[1]
    fused = torch.zeros(
        (num_blocks, KV_STRIDE), dtype=torch.uint8, device=kv_p.device
    )
    fused[:, :dp] = kv_p
    fused[:, dp : dp + ds] = kv_s
    return fused.contiguous()


def fp4_paged_mqa_logits(
    q_p,
    q_s,
    kv_cache_fp4,
    weights,
    out_logits,
    context_lens,
    block_tables,
    max_model_len,
    split_kv=None,
):
    """Paged MXFP4 MQA-logits (KV block_size == 1).

    Computes logits[b*N+n, j] = sum_h relu(Q[b,n,h,:] . K[tok(j),:]) * w[b*N+n,h]
    for j <= context_lens[b] - N + n, else -inf, with tok(j) = block_tables[b, j].

    q_p:          [batch, next_n, NUM_HEADS, HEAD_SIZE // 2],  uint8
    q_s:          [batch, next_n, NUM_HEADS, HEAD_SIZE // 32], uint8
    kv_cache_fp4: fused MXFP4 cache from `make_fp4_kv_cache`, [num_blocks, 80]
                  (also accepts [num_blocks, 1, 1, 80] and flattens it).
    weights:      [batch * next_n, NUM_HEADS], float32
    out_logits:   [batch * next_n, max_model_len], float32 (pre-filled with -inf)
    context_lens: [batch], int32
    block_tables: [batch, max_block_len], int32 (paged token indices)
    max_model_len: int
    split_kv:     optional kv-split factor (None -> graph-stable heuristic).

    Returns out_logits (written in place).
    """
    kv = kv_cache_fp4
    if kv.dim() > 2:
        kv = kv.reshape(kv.shape[0], -1)
    if not kv.is_contiguous():
        kv = kv.contiguous()
    batch, next_n = q_p.shape[0], q_p.shape[1]
    if split_kv is None:
        split_kv = pick_split_kv(batch, next_n)
    return _fp4_paged_mqa_logits(
        q_p,
        q_s,
        kv,
        block_tables,
        weights,
        context_lens,
        out_logits,
        int(max_model_len),
        int(split_kv),
    )
