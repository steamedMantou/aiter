# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Accuracy + performance comparison for the MXFP4 MQA-logits ops vs the FP8
"gluon" baselines, scored against an fp32 reference.

  * FP4 (aiter) : aiter.ops.fp4_mqa_logits / fp4_paged_mqa_logits  (HIP, gfx950)
  * FP8 (gluon) : aiter.ops.triton dense fp8_mqa_logits /
                  paged deepgemm_fp8_paged_mqa_logits

For each problem we draw one bf16 (q, kv) source and build:
  ref_bf16 : fp32 logits from the raw bf16 source            (the "truth")
  ref_fp4  : fp32 logits from the *dequantized MXFP4* source (FP4 kernel target)

We report calc_diff (1 - cosine-sim, deepgemm style) of each kernel vs ref_bf16
so FP4 and FP8 quantization error are directly comparable, hard-assert the FP4
kernel reproduces ref_fp4 (kernel correctness, independent of the FP8 baseline),
and time both kernels (CUDA-graph replay) to compare throughput.

The FP4 kernel is HEAD_SIZE == 128 only (its scaled-FP4 matrix core contracts
the full D=128), so every case uses D=128.

Run as a script for the full accuracy+perf table:
    python op_tests/test_fp4_mqa_logits.py
Or as correctness-only pytest:
    pytest op_tests/test_fp4_mqa_logits.py
"""
import random
from dataclasses import dataclass

import pytest
import torch

from aiter.ops.fp4_mqa_logits import (
    fp4_mqa_logits,
    fp4_paged_mqa_logits,
    make_fp4_kv_cache,
)
from aiter.ops.triton.attention.fp8_mqa_logits import fp8_mqa_logits
from aiter.ops.triton.attention.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits
from aiter.ops.triton.utils.types import get_fp8_dtypes, get_fp8_e4m3_dtype

e5m2_type, e4m3_type = get_fp8_dtypes()
_FP8_MAX = torch.finfo(e4m3_type).max


# ===========================================================================
# MXFP4 (E2M1 data + E8M0 micro-scale) quant / dequant -- matches the layout
# the HIP kernel decodes (lo nibble = even idx, hi = odd; one e8m0 per 32 elems).
# ===========================================================================
_FP4_E2M1_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def _ceil_to_ue8m0(x: torch.Tensor) -> torch.Tensor:
    bits = x.abs().float().view(torch.int)
    exp = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).bool().int()
    return (exp.clamp(1, 254) << 23).view(torch.float)


def _quantize_to_fp4_e2m1(x: torch.Tensor) -> torch.Tensor:
    ax = x.abs().clamp_max(6.0)
    boundaries = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=x.device, dtype=ax.dtype
    )
    idx = torch.bucketize(ax, boundaries)
    code = idx.to(torch.uint8)
    sign = (x < 0) & (idx != 0)
    code = code | (sign.to(torch.uint8) << 3)
    return code.view(torch.int8)


def _dequantize_from_fp4_e2m1(x: torch.Tensor) -> torch.Tensor:
    vals = torch.tensor(_FP4_E2M1_VALUES, device=x.device, dtype=torch.float)
    sign, value_idx = (x & 0x08) != 0, (x & 0x07).to(torch.int)
    value = vals[value_idx]
    return torch.where(sign & (value_idx != 0), -value, value)


def quant_mxfp4_rowwise(x: torch.Tensor, gran_k: int = 32):
    """[M, K] fp tensor -> (packed_uint8 [M, K//2], scale_uint8 [M, K//gran_k])."""
    assert x.dim() == 2 and x.size(1) % gran_k == 0
    m, n = x.shape
    x_view = x.view(m, n // gran_k, gran_k)
    x_amax = x_view.abs().float().amax(dim=2).clamp_min(1e-4)
    sf = _ceil_to_ue8m0(x_amax / 6.0)
    x_scaled = x_view * (1.0 / sf.unsqueeze(2))
    codes = _quantize_to_fp4_e2m1(x_scaled).view(m, n)
    codes2 = codes.view(m, n // 2, 2)
    packed = (codes2[:, :, 0] & 0x0F) | ((codes2[:, :, 1] & 0x0F) << 4)
    scale_e8m0 = (sf.view(torch.int) >> 23).to(torch.uint8)
    return packed.to(torch.uint8).contiguous(), scale_e8m0.contiguous()


def dequant_mxfp4_rowwise(packed, scale_e8m0, gran_k: int = 32) -> torch.Tensor:
    """Inverse of quant_mxfp4_rowwise -> fp32 [M, K]."""
    m, n2 = packed.shape
    n = n2 * 2
    unpacked = torch.zeros((m, n), dtype=torch.int8, device=packed.device)
    unpacked[:, ::2] = (packed & 0x0F).to(torch.int8)
    unpacked[:, 1::2] = ((packed >> 4) & 0x0F).to(torch.int8)
    deq = _dequantize_from_fp4_e2m1(unpacked)
    sf = (scale_e8m0.to(torch.int) << 23).view(torch.float)
    group_idx = torch.arange(n, device=packed.device) // gran_k
    return deq * sf[:, group_idx]


# ===========================================================================
# FP8 quant helpers (mirror aiter test_fp8_mqa_logits / bench_deepgemm)
# ===========================================================================
def _per_custom_dims_cast_to_fp8(x, dims):
    excluded = tuple(i for i in range(x.dim()) if i not in set(dims))
    x_amax = x.abs().float().amax(dim=excluded, keepdim=True).clamp(1e-4)
    sf = x_amax / _FP8_MAX
    return (x * (1.0 / sf)).to(e4m3_type), sf.squeeze()


def _kv_cache_cast_to_fp8(x):
    """[num_blocks, block, heads, dim] bf16 -> fused fp8 cache (data | f32 scale)."""
    fp8_dtype = get_fp8_e4m3_dtype()
    fp8_max = torch.finfo(fp8_dtype).max
    num_blocks, block_size, num_heads, head_dim = x.shape
    x_amax = x.abs().float().amax(dim=3, keepdim=True).clamp(1e-4)
    sf = x_amax / fp8_max
    x_scaled = (x * (1.0 / sf)).to(fp8_dtype)
    x_fp8 = torch.empty(
        (num_blocks, block_size * (head_dim + 4)), device=x.device, dtype=torch.uint8
    )
    x_fp8[:, : block_size * head_dim] = x_scaled.view(
        num_blocks, block_size * head_dim
    ).view(dtype=torch.uint8)
    x_fp8[:, block_size * head_dim : block_size * head_dim + 4 * block_size] = (
        sf.view(num_blocks, block_size).view(dtype=torch.uint8)
    )
    return x_fp8.view(num_blocks, block_size, num_heads, head_dim + 4)


# ===========================================================================
# fp32 references + scoring
# ===========================================================================
def calc_diff(x, y):
    x, y = x.double(), y.double()
    denom = (x * x + y * y).sum()
    return (1 - 2 * (x * y).sum() / denom).item()


@torch.no_grad()
def _ref_dense(q, kv, weights, ks, ke, kv_chunk=1024):
    M, H, D = q.shape
    Nkv = kv.shape[0]
    out = torch.full((M, Nkv), float("-inf"), device=q.device, dtype=torch.float32)
    ks_l, ke_l = ks.tolist(), ke.tolist()
    for m in range(M):
        a, b = ks_l[m], ke_l[m]
        if a >= b:
            continue
        qx, w = q[m], weights[m]
        for c0 in range(a, b, kv_chunk):
            c1 = min(b, c0 + kv_chunk)
            s = (qx[:, None, :] * kv[c0:c1][None, :, :]).sum(-1)  # [H,t]
            out[m, c0:c1] = (torch.relu(s) * w[:, None]).sum(0)
    return out


def _masked_dense(out, ref, ks, ke, Nkv):
    n = torch.arange(Nkv, device=out.device)
    mask = (n[None, :] >= ks[:, None]) & (n[None, :] < ke[:, None])
    return out.masked_fill(~mask, 0.0), ref.masked_fill(~mask, 0.0)


# ===========================================================================
# Paged problem (DeepSeek "lightning indexer" decode; block_size == 1)
# ===========================================================================
@dataclass
class PagedProblem:
    batch: int
    next_n: int
    heads: int
    dim: int
    max_model_len: int
    q: torch.Tensor
    kv: torch.Tensor
    weights: torch.Tensor
    context_lens: torch.Tensor
    block_tables: torch.Tensor
    q_p: torch.Tensor
    q_s: torch.Tensor
    kv_p: torch.Tensor
    kv_s: torch.Tensor
    q_deq: torch.Tensor
    kv_deq: torch.Tensor


def make_paged_problem(batch, next_n, heads, dim, avg_kv_length, var_ratio=0.5, seed=0):
    torch.manual_seed(seed)
    random.seed(seed)
    assert dim % 32 == 0
    device = "cuda"
    max_model_len = 2 * avg_kv_length
    num_blocks = max_model_len

    context_lens = torch.randint(
        int((1 - var_ratio) * avg_kv_length),
        int((1 + var_ratio) * avg_kv_length) + 1,
        (batch,),
        device=device,
    ).to(torch.int32)

    q = torch.randn((batch, next_n, heads, dim), device=device, dtype=torch.bfloat16)
    kv = torch.randn((num_blocks, dim), device=device, dtype=torch.bfloat16)
    weights = torch.randn((batch * next_n, heads), device=device, dtype=torch.float32)

    max_block_len = int(context_lens.max().item())
    block_tables = torch.zeros((batch, max_block_len), device=device, dtype=torch.int32)
    pool = torch.randperm(num_blocks, device=device, dtype=torch.int32)
    counter = 0
    for i in range(batch):
        ctx = int(context_lens[i].item())
        if ctx <= 0:
            continue
        idx = torch.arange(counter, counter + ctx, device=device) % num_blocks
        block_tables[i, :ctx] = pool[idx]
        counter += ctx

    q_p, q_s = quant_mxfp4_rowwise(q.reshape(-1, dim).float())
    q_p = q_p.view(batch, next_n, heads, dim // 2)
    q_s = q_s.view(batch, next_n, heads, dim // 32)
    q_deq = dequant_mxfp4_rowwise(
        q_p.reshape(-1, dim // 2), q_s.reshape(-1, dim // 32)
    ).view(batch, next_n, heads, dim)
    kv_p, kv_s = quant_mxfp4_rowwise(kv.float())
    kv_deq = dequant_mxfp4_rowwise(kv_p, kv_s)

    return PagedProblem(
        batch, next_n, heads, dim, max_model_len, q, kv, weights, context_lens,
        block_tables, q_p, q_s, kv_p, kv_s, q_deq, kv_deq,
    )


@torch.no_grad()
def ref_paged(p: PagedProblem, use_fp4=True, kv_chunk=8192):
    B, N, H, D = p.batch, p.next_n, p.heads, p.dim
    q = p.q_deq if use_fp4 else p.q.float()
    kv = p.kv_deq if use_fp4 else p.kv.float()
    out = torch.full((B * N, p.max_model_len), float("-inf"),
                     device=p.q.device, dtype=torch.float32)
    ctx_list = p.context_lens.tolist()
    for i in range(B):
        ctx = ctx_list[i]
        if ctx <= 0:
            continue
        toks = p.block_tables[i, :ctx].long()
        K = kv[toks]
        w = p.weights[i * N : (i + 1) * N]
        for n in range(N):
            qx = q[i, n]
            limit = ctx - N + n
            row = out[i * N + n]
            for c0 in range(0, ctx, kv_chunk):
                c1 = min(ctx, c0 + kv_chunk)
                s = (qx[:, None, :] * K[c0:c1][None, :, :]).sum(-1)
                s = (torch.relu(s) * w[n][:, None]).sum(0)
                j = torch.arange(c0, c1, device=s.device)
                row[c0:c1] = torch.where(j <= limit, s, torch.full_like(s, float("-inf")))
    return out


def masked_diff_paged(out_logits, ref, p: PagedProblem):
    mml = p.max_model_len
    positions = torch.arange(mml, device=out_logits.device).unsqueeze(0).expand(
        p.batch * p.next_n, -1
    )
    row_idx = torch.arange(p.batch * p.next_n, device=out_logits.device) // p.next_n
    nn_off = torch.arange(p.batch * p.next_n, device=out_logits.device) % p.next_n
    limits = (p.context_lens[row_idx] - p.next_n + nn_off).unsqueeze(1)
    mask = positions <= limits
    return calc_diff(out_logits.masked_fill(~mask, 0), ref.masked_fill(~mask, 0))


# ===========================================================================
# Timing: device-time per call via CUDA-graph replay (removes launch overhead).
# ===========================================================================
def bench_us(fn, iters=100, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    try:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        torch.cuda.synchronize()
        start.record()
        for _ in range(iters):
            g.replay()
        end.record()
        torch.cuda.synchronize()
    except Exception:
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1e3  # us


# ===========================================================================
# Dense comparison
# ===========================================================================
def _build_dense(M, H, D, Nkv, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(M, H, D, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(Nkv, D, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(M, H, device="cuda", dtype=torch.float32)
    ks = torch.zeros(M, dtype=torch.int32, device="cuda")
    ke = (torch.arange(M, dtype=torch.int32, device="cuda") + (Nkv - M) + 1).clamp_(1, Nkv)

    q_p, q_s = quant_mxfp4_rowwise(q.reshape(-1, D).float())
    q_p = q_p.view(M, H, D // 2).contiguous()
    q_s = q_s.view(M, H, D // 32).contiguous()
    q_deq = dequant_mxfp4_rowwise(q_p.reshape(-1, D // 2), q_s.reshape(-1, D // 32)).view(M, H, D)
    kv_p, kv_s = quant_mxfp4_rowwise(kv.float())
    kv_p, kv_s = kv_p.contiguous(), kv_s.contiguous()
    kv_deq = dequant_mxfp4_rowwise(kv_p, kv_s)
    return dict(q=q, kv=kv, weights=weights, ks=ks, ke=ke, q_p=q_p, q_s=q_s,
                kv_p=kv_p, kv_s=kv_s, q_deq=q_deq, kv_deq=kv_deq)


@torch.inference_mode()
def compare_dense(M, H, D, Nkv, do_perf=True, seed=0):
    c = _build_dense(M, H, D, Nkv, seed)
    ref_bf16 = _ref_dense(c["q"].float(), c["kv"].float(), c["weights"], c["ks"], c["ke"])
    ref_fp4 = _ref_dense(c["q_deq"], c["kv_deq"], c["weights"], c["ks"], c["ke"])

    out_fp4 = fp4_mqa_logits(c["q_p"], c["q_s"], c["kv_p"], c["kv_s"], c["weights"], c["ks"], c["ke"])
    torch.cuda.synchronize()
    q_fp8 = c["q"].to(e4m3_type)
    kv_fp8, scales = _per_custom_dims_cast_to_fp8(c["kv"], (0,))
    out_fp8 = fp8_mqa_logits(q_fp8, kv_fp8, scales, c["weights"], c["ks"], c["ke"], True)
    torch.cuda.synchronize()

    a4, rb = _masked_dense(out_fp4, ref_bf16, c["ks"], c["ke"], Nkv)
    a8, _ = _masked_dense(out_fp8, ref_bf16, c["ks"], c["ke"], Nkv)
    af4, rf4 = _masked_dense(out_fp4, ref_fp4, c["ks"], c["ke"], Nkv)
    d_fp4 = calc_diff(a4, rb)
    d_fp8 = calc_diff(a8, rb)
    d_fp4_fp4 = calc_diff(af4, rf4)

    n = torch.arange(Nkv, device=out_fp4.device)
    valid = (n[None, :] >= c["ks"][:, None]) & (n[None, :] < c["ke"][:, None])
    mask_ok = torch.equal(torch.isneginf(out_fp4), ~valid)

    fp4_us = fp8_us = None
    if do_perf:
        fp4_us = bench_us(lambda: fp4_mqa_logits(
            c["q_p"], c["q_s"], c["kv_p"], c["kv_s"], c["weights"], c["ks"], c["ke"]))
        fp8_us = bench_us(lambda: fp8_mqa_logits(
            q_fp8, kv_fp8, scales, c["weights"], c["ks"], c["ke"], True))

    return dict(d_fp4=d_fp4, d_fp8=d_fp8, d_fp4_fp4=d_fp4_fp4, mask_ok=mask_ok,
                fp4_us=fp4_us, fp8_us=fp8_us)


# ===========================================================================
# Paged comparison
# ===========================================================================
def _run_gluon_fp8_paged(p, out, chunk_k=128, wpe=5):
    q_fp8 = p.q.to(get_fp8_e4m3_dtype())
    kv_fp8 = _kv_cache_cast_to_fp8(p.kv.view(p.max_model_len, 1, 1, p.dim))

    def call():
        return deepgemm_fp8_paged_mqa_logits(
            q_fp8, kv_fp8, p.weights, out, p.context_lens, p.block_tables,
            p.max_model_len, Preshuffle=False, KVBlockSize=1, ChunkK=chunk_k, WavePerEU=wpe)

    return call


@torch.inference_mode()
def compare_paged(B, N, H, D, kv_length, do_perf=True, seed=0):
    p = make_paged_problem(B, N, H, D, kv_length, seed=seed)
    ref_bf16 = ref_paged(p, use_fp4=False)
    ref_fp4 = ref_paged(p, use_fp4=True)

    kv_cache = make_fp4_kv_cache(p.kv_p, p.kv_s)
    out_fp4 = torch.full((B * N, p.max_model_len), float("-inf"), device="cuda", dtype=torch.float32)

    def fp4_call():
        return fp4_paged_mqa_logits(p.q_p, p.q_s, kv_cache, p.weights, out_fp4,
                                    p.context_lens, p.block_tables, p.max_model_len)

    fp4_call()
    torch.cuda.synchronize()
    d_fp4 = masked_diff_paged(out_fp4, ref_bf16, p)
    d_fp4_fp4 = masked_diff_paged(out_fp4, ref_fp4, p)

    out_fp8 = torch.full((B * N, p.max_model_len), float("-inf"), device="cuda", dtype=torch.float32)
    d_fp8 = None
    fp8_call = None
    try:
        fp8_call = _run_gluon_fp8_paged(p, out_fp8)
        fp8_call()
        torch.cuda.synchronize()
        d_fp8 = masked_diff_paged(out_fp8, ref_bf16, p)
    except Exception as e:
        fp8_call = None
        d_fp8 = repr(e)[:48]

    fp4_us = fp8_us = None
    if do_perf:
        fp4_us = bench_us(fp4_call)
        if fp8_call is not None:
            try:
                fp8_us = bench_us(fp8_call)
            except Exception:
                fp8_us = None

    return dict(d_fp4=d_fp4, d_fp8=d_fp8, d_fp4_fp4=d_fp4_fp4, fp4_us=fp4_us,
                fp8_us=fp8_us, ctx_sum=int(p.context_lens.sum()))


# ===========================================================================
# pytest correctness gates (FP4 must reproduce its fp4 dequant reference)
# ===========================================================================
@pytest.mark.parametrize(
    "M, Nkv", [(128, 2048), (256, 2048), (64, 4096), (333, 1560), (16, 8192)]
)
@pytest.mark.parametrize("H", [64])
@torch.inference_mode()
def test_fp4_mqa_logits_dense(M, H, Nkv):
    r = compare_dense(M, H, 128, Nkv, do_perf=False)
    assert r["mask_ok"], "FP4 -inf mask mismatch"
    assert r["d_fp4_fp4"] < 1e-3, f"FP4 vs fp4-ref too large: {r['d_fp4_fp4']}"
    assert r["d_fp4"] < 0.3, f"FP4 vs bf16 unexpectedly large: {r['d_fp4']}"


@pytest.mark.parametrize(
    "B, N, kv_length", [(1, 1, 2048), (2, 2, 4096), (2, 1, 8192), (4, 1, 16384)]
)
@pytest.mark.parametrize("H", [64])
@torch.inference_mode()
def test_fp4_paged_mqa_logits(B, N, H, kv_length):
    r = compare_paged(B, N, H, 128, kv_length, do_perf=False)
    assert r["d_fp4_fp4"] < 1e-3, f"FP4 paged vs fp4-ref too large: {r['d_fp4_fp4']}"
    assert r["d_fp4"] < 0.3, f"FP4 paged vs bf16 unexpectedly large: {r['d_fp4']}"


# ===========================================================================
# Script entry: full accuracy + performance comparison table
# ===========================================================================
def _fmt_us(x):
    return f"{x:8.1f}" if isinstance(x, float) else f"{str(x):>8}"


def _fmt_speedup(fp8_us, fp4_us):
    if isinstance(fp8_us, float) and isinstance(fp4_us, float) and fp4_us > 0:
        return f"{fp8_us / fp4_us:.2f}x"
    return "-"


def main():
    torch.cuda.init()
    print("== DENSE: aiter FP4  vs  FP8 gluon (fp8_mqa_logits)  vs  fp32 ref ==")
    print(f"{'M':>4} {'H':>3} {'Nkv':>6} | {'FP4/bf16':>9} {'FP8/bf16':>9} {'FP4/fp4ref':>10} "
          f"| {'FP4 us':>8} {'FP8 us':>8} {'FP8/FP4':>7}")
    for (M, H, Nkv) in [(128, 64, 2048), (256, 64, 2048), (64, 64, 4096),
                        (333, 64, 1560), (16, 64, 8192), (2048, 64, 8192)]:
        r = compare_dense(M, H, 128, Nkv)
        print(f"{M:>4} {H:>3} {Nkv:>6} | {r['d_fp4']:9.2e} {r['d_fp8']:9.2e} "
              f"{r['d_fp4_fp4']:10.1e} | {_fmt_us(r['fp4_us'])} {_fmt_us(r['fp8_us'])} "
              f"{_fmt_speedup(r['fp8_us'], r['fp4_us']):>7}")

    print("\n== PAGED: aiter FP4  vs  FP8 gluon (deepgemm_fp8_paged)  vs  fp32 ref ==")
    print(f"{'B':>2} {'N':>2} {'H':>3} {'kv':>6} | {'FP4/bf16':>9} {'FP8/bf16':>9} "
          f"{'FP4/fp4ref':>10} | {'FP4 us':>8} {'FP8 us':>8} {'FP8/FP4':>7}")
    for (B, N, H, L) in [(1, 1, 64, 2048), (2, 2, 64, 4096), (2, 1, 64, 8192),
                         (4, 1, 64, 16384), (8, 1, 64, 32768)]:
        r = compare_paged(B, N, H, 128, L)
        d8 = f"{r['d_fp8']:9.2e}" if isinstance(r["d_fp8"], float) else f"{str(r['d_fp8'])[:9]:>9}"
        print(f"{B:>2} {N:>2} {H:>3} {L:>6} | {r['d_fp4']:9.2e} {d8} "
              f"{r['d_fp4_fp4']:10.1e} | {_fmt_us(r['fp4_us'])} {_fmt_us(r['fp8_us'])} "
              f"{_fmt_speedup(r['fp8_us'], r['fp4_us']):>7}")

    print("\nFP4/bf16, FP8/bf16 = quantization error vs fp32 truth (calc_diff, lower=better)")
    print("FP4/fp4ref = FP4 kernel vs its exact fp4 dequant target (~0 => correct)")
    print("FP8/FP4 us ratio > 1 => FP4 is faster.")


if __name__ == "__main__":
    main()
