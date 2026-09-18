# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""DSV4 sparse MLA prefill at the shapes vLLM actually launches, ISL=100k OSL=1.

A 102400-token prompt is chunked into 13 x 8192, and each rank's slice of a
chunk depends on how the model is parallelized:

  TP=8  (tp-prod) splits the 128 heads   -> H=16,  T=8192 per rank
  PCP=8 (TP=1)    splits the sequence    -> H=128, T=1024 per rank

Both do 16 * 8192 == 128 * 1024 head-tokens, so the two rows are the same total
work arranged differently, which is exactly what decides whether the assembly
kernel beats Triton.

Compares the production Triton kernel from vLLM against aiter's Triton kernel,
the cherry-picked dsv4_mla_prefill assembly kernel, and the opus paged kernel.
"""

from __future__ import annotations

import math
import time

import torch

from aiter.ops.dsv4_mla_prefill import dsv4_mla_prefill
from aiter.ops.triton.attention.sparse_attention_dsv4 import sparse_mla_fwd_dsv4
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    _rocm_sparse_attn_prefill_ragged_triton,
)

from bench_dsv4_mla_vs_triton_isl100k import (  # noqa: E402
    D_HEAD,
    D_NOPE,
    D_ROPE,
    Pool,
    _bench,
)

SWA, TOPK = 128, 1024
DEV = "cuda"

# (label, heads per rank, tokens per rank per chunk, chunks per 100k prompt).
# Total work is held at 128*1024 head-tokens so the rows differ only in shape.
CONFIGS = [
    ("TP=8            (heads split)", 16, 8192, 13),
    ("TP=4 x PCP=2", 32, 4096, 13),
    ("TP=2 x PCP=4", 64, 2048, 13),
    ("PCP=8      (sequence split)", 128, 1024, 13),
]


def run(label: str, h: int, t: int, chunks: int, layers: int = 61, nnz: int | None = None) -> dict:
    nnz = nnz or TOPK + SWA
    pool_tokens = 32768
    pool = Pool(pool_tokens)
    g = torch.Generator(device=DEV).manual_seed(0)
    q_nope = (torch.randn(t, h, D_NOPE, device=DEV, generator=g) * 0.125).to(torch.bfloat16)
    q_rope = (torch.randn(t, h, D_ROPE, device=DEV, generator=g) * 0.125).to(torch.bfloat16)
    q = torch.cat([q_nope, q_rope], dim=-1)
    sink = torch.randn(h, dtype=torch.float32, device=DEV, generator=g) * 0.25
    scale = 1.0 / math.sqrt(D_HEAD)
    idx = torch.randint(0, pool.num_tokens, (t * nnz,), dtype=torch.int32, device=DEV)
    indptr = torch.arange(0, (t + 1) * nnz, nnz, dtype=torch.int32, device=DEV)
    empty_idx = torch.zeros(0, dtype=torch.int32, device=DEV)
    empty_ptr = torch.zeros(t + 1, dtype=torch.int32, device=DEV)
    nope_v, rope_v = pool.views()
    kv_bf16 = pool.decode_bf16()
    max_e = torch.zeros(1, dtype=torch.int32, device=DEV)
    out_mla = torch.empty(t, h, D_HEAD, dtype=torch.bfloat16, device=DEV)
    out_aiter_tri = torch.empty_like(q)
    out_opus = torch.empty_like(out_mla)

    def vllm_triton():
        _rocm_sparse_attn_prefill_ragged_triton(
            q=q, kv=kv_bf16, indices=idx, indptr=indptr, scale=scale,
            attn_sink=sink, nope_head_dim=D_NOPE, rope_head_dim=D_ROPE,
        )

    def aiter_triton():
        sparse_mla_fwd_dsv4(
            q=q, kv=kv_bf16, kv_indices=idx, kv_indptr=indptr,
            softmax_scale=scale, attn_sink=sink, out=out_aiter_tri,
        )

    def mla_asm():
        dsv4_mla_prefill(
            q_nope=q_nope, q_rope=q_rope,
            unified_kv_nope=nope_v, unified_kv_rope=rope_v,
            kv_indices_prefix=idx, kv_indptr_prefix=indptr,
            kv_nope=nope_v, kv_rope=rope_v,
            kv_indices_extend=empty_idx, kv_indptr_extend=empty_ptr,
            attn_sink=sink, kv_max_e=max_e, softmax_scale=scale, out=out_mla,
            page_shift_prefix=pool.page_shift,
            rows_per_page_prefix=pool.rows_per_page,
            scale_off_prefix=pool.scale_off,
            page_shift_extend=pool.page_shift,
            rows_per_page_extend=pool.rows_per_page,
            scale_off_extend=pool.scale_off,
        )

    def opus_paged():
        from aiter.ops.pa_sparse_prefill_opus import pa_sparse_prefill_fp8_opus_paged

        pa_sparse_prefill_fp8_opus_paged(
            q_nope, q_rope, nope_v, rope_v, idx, indptr, nope_v, rope_v,
            empty_idx, empty_ptr, sink, scale,
            pool.page_shift, pool.rows_per_page, pool.scale_off,
            pool.page_shift, pool.rows_per_page, pool.scale_off, out=out_opus,
        )

    impls = [
        ("vLLM Triton (production)", vllm_triton),
        ("aiter Triton", aiter_triton),
        ("dsv4_mla_prefill (ASM)", mla_asm),
        ("opus fp8 paged", opus_paged),
    ]

    print(f"\n=== {label}: H={h} T={t} nnz/row={nnz} ===")
    flops = 4.0 * h * D_HEAD * t * nnz
    results: dict[str, float] = {}
    errs: dict[str, float] = {}
    ref_out = None
    for name, fn in impls:
        try:
            fn()
            torch.cuda.synchronize()
            results[name] = _bench(fn)
        except Exception as e:
            print(f"  {name:26s} failed: {type(e).__name__}: {e}")
            continue
        got = {
            "vLLM Triton (production)": None,
            "aiter Triton": out_aiter_tri,
            "dsv4_mla_prefill (ASM)": out_mla,
            "opus fp8 paged": out_opus,
        }[name]
        if name == "vLLM Triton (production)":
            ref_out = vllm_triton_out(q, kv_bf16, idx, indptr, scale, sink)
        elif ref_out is not None:
            errs[name] = (got.float() - ref_out.float()).abs().mean().item()

    ref = results.get("vLLM Triton (production)")
    # 61 layers x 13 chunks of sparse attention make up one 100k prefill.
    calls = layers * chunks
    print(f"  {'impl':26s} {'ms/call':>9} {'TFLOPS':>9} {'vs vLLM':>9} "
          f"{'per 100k prefill':>17} {'mean|err|':>10}")
    for name, ms in results.items():
        sp = f"{ref / ms:.2f}x" if ref else "-"
        err = f"{errs[name]:.4f}" if name in errs else "-"
        print(f"  {name:26s} {ms:9.3f} {flops / (ms * 1e-3) / 1e12:9.1f} {sp:>9} "
              f"{ms * calls / 1000:15.2f} s {err:>10}")
    return results


def vllm_triton_out(q, kv, idx, indptr, scale, sink):
    return _rocm_sparse_attn_prefill_ragged_triton(
        q=q, kv=kv, indices=idx, indptr=indptr, scale=scale,
        attn_sink=sink, nope_head_dim=D_NOPE, rope_head_dim=D_ROPE,
    )


def main() -> None:
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--json", dest="json_out")
    args = ap.parse_args()

    print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).gcnArchName)
    t0 = time.time()
    out = []
    for label, h, t, chunks in CONFIGS:
        res = run(label, h, t, chunks)
        out.append({"label": label, "heads": h, "tokens": t, "ms": res})
    print(f"\nwall {time.time() - t0:.1f}s")
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
