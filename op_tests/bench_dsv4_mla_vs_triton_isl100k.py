# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Single-op DSV4 MLA prefill vs Triton at ISL=102400, OSL=1, bs=1.

Mirrors the production chunked-prefill launch: one 8192-token chunk at the
end of a 102400-token prompt, topk=1024 + SWA=128, page_size=256.
"""

from __future__ import annotations

import math
import time

import torch

from aiter.ops.dsv4_mla_prefill import dsv4_mla_prefill
from aiter.ops.triton.attention.sparse_attention_dsv4 import sparse_mla_fwd_dsv4

D_NOPE, D_ROPE, D_HEAD = 448, 64, 512
ROW, NBLK, SCALE_SLOT = 576, 7, 8
ISL, CHUNK, TOPK, SWA = 102400, 8192, 1024, 128
PAGE_SIZE = 256
DEV = "cuda"


class Pool:
    def __init__(self, num_tokens: int, page_size: int = PAGE_SIZE, seed: int = 0):
        self.page_size = page_size
        self.page_shift = page_size.bit_length() - 1
        self.num_pages = (num_tokens + page_size - 1) // page_size
        self.num_tokens = self.num_pages * page_size
        self.bytes_per_page = -(-page_size * (ROW + SCALE_SLOT) // ROW) * ROW
        self.rows_per_page = self.bytes_per_page // ROW
        self.scale_off = page_size * ROW
        g = torch.Generator(device=DEV).manual_seed(seed)
        real = torch.randn(self.num_tokens, D_NOPE, device=DEV, generator=g) * 0.5
        blk = real.reshape(self.num_tokens, NBLK, 64)
        e = torch.ceil(
            torch.log2(blk.abs().amax(-1).clamp(min=1e-30) / 448.0)
        ).to(torch.int32)
        self.exp_b = (e + 127).clamp(1, 200).to(torch.uint8)
        q8 = (
            blk / torch.exp2((self.exp_b.int() - 127).float()).unsqueeze(-1)
        ).to(torch.float8_e4m3fn)
        self.rope = (
            torch.randn(self.num_tokens, D_ROPE, device=DEV, generator=g) * 0.5
        ).to(torch.bfloat16)
        self.buf = torch.zeros(
            self.num_pages, self.bytes_per_page, dtype=torch.uint8, device=DEV
        )
        nope_b = q8.reshape(self.num_tokens, D_NOPE).view(torch.uint8)
        rope_b = self.rope.view(torch.uint8).reshape(self.num_tokens, 128)
        region_a = torch.zeros(self.num_pages, page_size, ROW, dtype=torch.uint8, device=DEV)
        region_a[:, :, :D_NOPE] = nope_b.reshape(self.num_pages, page_size, D_NOPE)
        region_a[:, :, D_NOPE:] = rope_b.reshape(self.num_pages, page_size, 128)
        self.buf[:, : page_size * ROW] = region_a.reshape(self.num_pages, page_size * ROW)
        scale_tail = self.buf[:, self.scale_off : self.scale_off + page_size * SCALE_SLOT]
        scale_tail = scale_tail.reshape(self.num_pages, page_size, SCALE_SLOT)
        scale_tail[:, :, :NBLK] = self.exp_b.reshape(self.num_pages, page_size, NBLK)

    def views(self):
        grid = self.num_pages * self.rows_per_page
        flat = self.buf.view(-1)
        nope = torch.as_strided(flat.view(torch.float8_e4m3fn), (grid, 512), (ROW, 1))
        rope = torch.as_strided(
            flat.view(torch.bfloat16),
            (grid, 64),
            (ROW // 2, 1),
            storage_offset=D_NOPE // 2,
        )
        return nope, rope

    def decode_bf16(self) -> torch.Tensor:
        ps, npg = self.page_size, self.num_pages
        region_a = self.buf[:, : ps * ROW].reshape(npg, ps, ROW)
        nope8 = region_a[:, :, :D_NOPE].reshape(-1, D_NOPE).view(torch.float8_e4m3fn)
        sc = (
            self.buf[:, self.scale_off : self.scale_off + ps * SCALE_SLOT]
            .reshape(npg, ps, SCALE_SLOT)[:, :, :NBLK]
            .reshape(-1, NBLK)
            .int()
        )
        nope = nope8.float() * torch.exp2((sc - 127).float()).repeat_interleave(
            64, dim=-1
        )
        return torch.cat([nope.to(torch.bfloat16), self.rope], dim=-1)


def _bench(fn, warmup=5, reps=20) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(reps):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / reps


def run(h: int, t: int = CHUNK, nnz: int = TOPK + SWA, pool_tokens: int | None = None):
    pool_tokens = pool_tokens or min(ISL // 4 + t + SWA, 32768)
    print(f"\n=== H={h} T={t} nnz/row={nnz} pool_tokens~{pool_tokens} page={PAGE_SIZE} ===")
    pool = Pool(pool_tokens)
    print(f"pool tokens={pool.num_tokens} pages={pool.num_pages} rows/page={pool.rows_per_page}")
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
    out_triton = torch.empty_like(q)

    def mla():
        dsv4_mla_prefill(
            q_nope=q_nope,
            q_rope=q_rope,
            unified_kv_nope=nope_v,
            unified_kv_rope=rope_v,
            kv_indices_prefix=idx,
            kv_indptr_prefix=indptr,
            kv_nope=nope_v,
            kv_rope=rope_v,
            kv_indices_extend=empty_idx,
            kv_indptr_extend=empty_ptr,
            attn_sink=sink,
            kv_max_e=max_e,
            softmax_scale=scale,
            out=out_mla,
            page_shift_prefix=pool.page_shift,
            rows_per_page_prefix=pool.rows_per_page,
            scale_off_prefix=pool.scale_off,
            page_shift_extend=pool.page_shift,
            rows_per_page_extend=pool.rows_per_page,
            scale_off_extend=pool.scale_off,
        )

    def triton():
        sparse_mla_fwd_dsv4(
            q=q,
            kv=kv_bf16,
            kv_indices=idx,
            kv_indptr=indptr,
            softmax_scale=scale,
            attn_sink=sink,
            out=out_triton,
        )

    # correctness on a short slice
    mla()
    triton()
    torch.cuda.synchronize()
    diff = (out_mla.float() - out_triton.float()).abs().mean().item()
    print(f"mean |MLA-Triton| = {diff:.4f}  (fp8 vs bf16, not expected to be 0)")

    ms_triton = _bench(triton)
    ms_mla = _bench(mla)
    ms_paged = None
    if h > 32:
        from aiter.ops.pa_sparse_prefill_opus import pa_sparse_prefill_fp8_opus_paged

        out_p = torch.empty_like(out_mla)

        def paged():
            pa_sparse_prefill_fp8_opus_paged(
                q_nope,
                q_rope,
                nope_v,
                rope_v,
                idx,
                indptr,
                nope_v,
                rope_v,
                empty_idx,
                empty_ptr,
                sink,
                scale,
                pool.page_shift,
                pool.rows_per_page,
                pool.scale_off,
                pool.page_shift,
                pool.rows_per_page,
                pool.scale_off,
                out=out_p,
            )

        try:
            paged()
            torch.cuda.synchronize()
            ms_paged = _bench(paged)
        except Exception as e:
            print(f"paged opus failed: {type(e).__name__}: {e}")

    flops = 4.0 * h * D_HEAD * t * nnz
    def tflops(ms):
        return flops / (ms * 1e-3) / 1e12

    print(f"Triton sparse MLA : {ms_triton:7.3f} ms  {tflops(ms_triton):6.1f} TFLOPS")
    print(f"dsv4_mla_prefill  : {ms_mla:7.3f} ms  {tflops(ms_mla):6.1f} TFLOPS  speedup {ms_triton/ms_mla:.2f}x")
    if ms_paged is not None:
        print(f"opus fp8 paged    : {ms_paged:7.3f} ms  {tflops(ms_paged):6.1f} TFLOPS  speedup {ms_triton/ms_paged:.2f}x")
    return ms_triton, ms_mla, ms_paged


def main():
    print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).gcnArchName)
    print(f"workload: ISL={ISL} OSL=1 bs=1 chunk={CHUNK} (one last prefill chunk)")
    t0 = time.time()
    run(h=16)
    run(h=128)
    print(f"\nwall {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
