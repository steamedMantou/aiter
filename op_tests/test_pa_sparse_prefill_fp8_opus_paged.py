#!/usr/bin/env python3
"""Accuracy of the paged opus fp8 prefill against vLLM's Triton kernel.

Builds a synthetic vLLM `fp8_ds_mla` pool (the real layout, from
vllm/models/deepseek_v4/common/ops/cache_utils.py) and runs three arms on the
same data:

  triton   vLLM's Triton sparse prefill on the *unquantised* bf16 KV
  flat     aiter's existing flat fp8 opus op, fed the same fp8 bytes with each
           per-64 exponent duplicated into its two per-32 slots, and a Q
           pre-packed per 32
  paged    the new op: reads the pool directly and packs Q itself, per head

The two arms now see identical KV but differently quantised Q -- per head
against per 32 -- so `paged vs flat` is no longer expected to be zero.  It
measures exactly what the coarser Q exponent costs; `paged vs triton` next to
`flat vs triton` says whether that shows up at all against the fp8 floor.
"""
import argparse, math, statistics
import torch

from aiter.ops.pa_sparse_prefill_opus import (
    pa_sparse_prefill_fp8_opus, pa_sparse_prefill_fp8_opus_paged)
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import rocm_sparse_attn_prefill

D, D_NOPE, D_ROPE, D_PAD = 512, 448, 64, 512
ROW_BYTES, SCALE_SLOT, QBLK = 576, 8, 64
NSCALE = D_NOPE // QBLK          # 7
FP8_MAX = 448.0


def _skip_if_unsupported() -> bool:
    if not torch.cuda.is_available():
        print("SKIP: no CUDA/HIP device"); return True
    arch = str(getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")).split(":")[0]
    if arch != "gfx950":
        print(f"SKIP: needs gfx950, found {arch}"); return True
    return False


def quant_per64(x):
    """vLLM's quantize_and_insert_k_kernel, verbatim: UE8M0, stored exp+127."""
    r = x.shape[0]
    b = x.float().reshape(r, NSCALE, QBLK)
    amax = b.abs().amax(-1).clamp(min=1e-4)
    e = torch.ceil(torch.log2(amax / FP8_MAX))
    enc = (e + 127.0).clamp(0, 255).to(torch.uint8)          # [r, 7]
    s = torch.exp2((enc.int() - 127).float()).unsqueeze(-1)
    q = torch.clamp(b / s, -FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    deq = (q.float() * s).reshape(r, D_NOPE)
    return q.reshape(r, D_NOPE), enc, deq


def build_pool(nope_fp8, enc, rope_bf16, page_size):
    """Lay the rows out exactly as vLLM's paged fp8_ds_mla cache does."""
    rows = nope_fp8.shape[0]
    num_pages = (rows + page_size - 1) // page_size
    # The pool pads bytes_per_page up to a multiple of the row size.  Using
    # page_size*576 instead leaves the scale region zero bytes long and every
    # scale write lands in the next page's fp8 data.
    bytes_per_page = -(-page_size * (ROW_BYTES + SCALE_SLOT) // ROW_BYTES) * ROW_BYTES
    rows_per_page = bytes_per_page // ROW_BYTES
    scale_off = page_size * ROW_BYTES

    pool = torch.zeros(num_pages, bytes_per_page, dtype=torch.uint8, device=nope_fp8.device)
    nope_u8 = nope_fp8.view(torch.uint8)
    rope_u8 = rope_bf16.view(torch.uint8).reshape(rows, D_ROPE * 2)
    for g in range(rows):
        p, j = g // page_size, g % page_size
        pool[p, j * ROW_BYTES : j * ROW_BYTES + D_NOPE] = nope_u8[g]
        pool[p, j * ROW_BYTES + D_NOPE : (j + 1) * ROW_BYTES] = rope_u8[g]
        pool[p, scale_off + j * SCALE_SLOT : scale_off + j * SCALE_SLOT + NSCALE] = enc[g]

    view_rows = num_pages * rows_per_page
    flat = pool.reshape(-1)
    nope_view = flat.view(torch.float8_e4m3fn).as_strided((view_rows, D_PAD), (ROW_BYTES, 1))
    rope_view = flat.view(torch.bfloat16).as_strided(
        (view_rows, D_ROPE), (ROW_BYTES // 2, 1), storage_offset=D_NOPE // 2)
    geom = dict(page_shift=page_size.bit_length() - 1,
                rows_per_page=rows_per_page, scale_off=scale_off)
    return pool, nope_view, rope_view, geom


def pack_flat(nope_fp8, enc):
    """The flat op's row: 448 fp8 | 14 per-32 E8M0 | pad, each per-64 byte twice."""
    r = nope_fp8.shape[0]
    out = torch.zeros(r, D_PAD, dtype=torch.uint8, device=nope_fp8.device)
    out[:, :D_NOPE] = nope_fp8.view(torch.uint8)
    out[:, D_NOPE : D_NOPE + 2 * NSCALE] = enc.repeat_interleave(2, dim=1)
    return out.view(torch.float8_e4m3fn)


def pack_q(src):
    """Q keeps the flat op's per-32 packing (FUSED_Q is not on yet)."""
    r, nblk = src.shape[0], D_NOPE // 32
    blk = src.float().reshape(r, nblk, 32)
    amax = blk.abs().amax(-1)
    e = torch.ceil(torch.log2(amax.clamp(min=1e-30) / FP8_MAX)).to(torch.int32)
    e = torch.where(amax == 0, torch.zeros_like(e), e)
    enc = (e + 127).clamp(1, 254).to(torch.uint8)
    v = (blk / torch.exp2((enc.int() - 127).float()).unsqueeze(-1)).to(torch.float8_e4m3fn)
    out = torch.zeros(r, D_PAD, dtype=torch.uint8, device=src.device)
    out[:, :D_NOPE] = v.reshape(r, D_NOPE).view(torch.uint8)
    out[:, D_NOPE : D_NOPE + nblk] = enc
    return out.view(torch.float8_e4m3fn)


def rel_l2(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30))


def run(T, H, rows, plen, elen, page_size, seed):
    dev = torch.device("cuda")
    torch.manual_seed(seed)
    q = (torch.randn(T, H, D, device=dev) * .5).to(torch.bfloat16)
    kv = (torch.randn(rows, D, device=dev) * .5).to(torch.bfloat16)
    sink = torch.randn(H, dtype=torch.float32, device=dev) * .25
    scale = 1.0 / math.sqrt(D)

    nope_fp8, enc, deq = quant_per64(kv[:, :D_NOPE])
    rope = kv[:, D_NOPE:].contiguous()
    pool, nope_view, rope_view, geom = build_pool(nope_fp8, enc, rope, page_size)
    flat_nope = pack_flat(nope_fp8, enc)

    p_ix = torch.randint(0, rows, (T, plen), dtype=torch.int32, device=dev)
    e_ix = torch.randint(0, rows, (T, elen), dtype=torch.int32, device=dev)
    p_ip = torch.arange(0, (T + 1) * plen, plen, dtype=torch.int32, device=dev)
    e_ip = torch.arange(0, (T + 1) * elen, elen, dtype=torch.int32, device=dev)

    q_nope = pack_q(q[..., :D_NOPE].reshape(-1, D_NOPE)).reshape(T, H, D_PAD)
    q_rope = q[..., D_NOPE:].contiguous()

    # A [..., :448] slice of the [T, H, 512] Q, row stride 512 -- this is what
    # vLLM hands over, and requiring a contiguous 448 here would force a copy.
    q_nope_bf16 = q[..., :D_NOPE]
    assert q_nope_bf16.stride(1) == D, "the sliced-Q case must keep the wide row stride"
    args = (q_nope_bf16, q_rope, nope_view, rope_view, p_ix.reshape(-1), p_ip,
            nope_view, rope_view, e_ix.reshape(-1), e_ip, sink, scale,
            geom["page_shift"], geom["rows_per_page"], geom["scale_off"],
            geom["page_shift"], geom["rows_per_page"], geom["scale_off"])
    out_paged = pa_sparse_prefill_fp8_opus_paged(*args)

    # The dense form: the same indices read as [T, topk] + lengths, which is the
    # shape vLLM already has.  It must agree with CSR bit for bit.
    p_len = torch.full((T,), plen, dtype=torch.int32, device=dev)
    e_len = torch.full((T,), elen, dtype=torch.int32, device=dev)
    out_dense = pa_sparse_prefill_fp8_opus_paged(
        *args, kv_lens_prefix=p_len, kv_lens_extend=e_len,
        kv_stride_q_prefix=plen, kv_stride_q_extend=elen)

    out_flat = pa_sparse_prefill_fp8_opus(
        q_nope, q_rope, flat_nope, rope, p_ix.reshape(-1), p_ip,
        flat_nope, rope, e_ix.reshape(-1), e_ip, sink, scale)

    tri_ix = torch.cat((p_ix, e_ix), dim=1)
    tri_len = torch.full((T,), plen + elen, dtype=torch.int32, device=dev)
    out_tri = torch.empty(T, H, D, dtype=torch.bfloat16, device=dev)
    rocm_sparse_attn_prefill(q=q, kv=kv.unsqueeze(1), indices=tri_ix, topk_length=tri_len,
                             scale=scale, head_dim=D, nope_head_dim=D_NOPE,
                             rope_head_dim=D_ROPE, attn_sink=sink, output=out_tri)
    # Triton on the dequantised KV isolates the kernel from the quantisation.
    kv_deq = torch.cat((deq.to(torch.bfloat16), rope), dim=1)
    out_tri_deq = torch.empty_like(out_tri)
    rocm_sparse_attn_prefill(q=q, kv=kv_deq.unsqueeze(1), indices=tri_ix, topk_length=tri_len,
                             scale=scale, head_dim=D, nope_head_dim=D_NOPE,
                             rope_head_dim=D_ROPE, attn_sink=sink, output=out_tri_deq)
    torch.cuda.synchronize()

    nf = int((~torch.isfinite(out_paged)).sum())
    return dict(nonfinite=nf,
                dense_eq=bool(torch.equal(out_paged, out_dense)),
                paged_vs_flat=rel_l2(out_paged, out_flat),
                exact=bool(torch.equal(out_paged, out_flat)),
                paged_vs_triton=rel_l2(out_paged, out_tri),
                flat_vs_triton=rel_l2(out_flat, out_tri),
                paged_vs_triton_deq=rel_l2(out_paged, out_tri_deq))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--page-size", type=int, default=64)
    a = ap.parse_args()
    cases = [("hca-4k", 1024, 128, 128, 32), ("csa", 1024, 128, 128, 512),
             ("hca-128k", 512, 128, 128, 1024), ("H=64", 512, 64, 128, 512),
             ("H=192", 256, 192, 128, 256), ("empty-extend", 512, 128, 256, 0)]
    print(f"page_size={a.page_size}  gpu={torch.cuda.get_device_name()}")
    print(f"{'case':14}{'T':>6}{'H':>5}{'rows/tok':>9}{'nonfin':>8}"
          f"{'paged vs flat':>15}{'dense==csr':>12}{'paged vs triton':>17}{'flat vs triton':>16}"
          f"{'vs triton(deq)':>16}")
    bad = 0
    for name, T, H, plen, elen in cases:
        if elen == 0:
            r = run(T, H, 8192, plen, 1, a.page_size, 20260821)   # 1 = minimal extend
        else:
            r = run(T, H, 8192, plen, elen, a.page_size, 20260821)
        # the bar is Triton: the fused Q pack must not move the fp8 cost
        flag = ("" if (r["nonfinite"] == 0 and r["dense_eq"]
                       and abs(r["paged_vs_triton"] - r["flat_vs_triton"]) < 2e-3)
                else "   <-- FAIL")
        if flag: bad += 1
        print(f"{name:14}{T:>6}{H:>5}{plen+elen:>9}{r['nonfinite']:>8}"
              f"{r['paged_vs_flat']:>15.3e}{str(r['dense_eq']):>12}{r['paged_vs_triton']:>17.3e}"
              f"{r['flat_vs_triton']:>16.3e}{r['paged_vs_triton_deq']:>16.3e}{flag}", flush=True)
    print("\nFAILURES:" if bad else "\nall cases within 2e-3 of the flat arm against Triton", bad or "")


if __name__ == "__main__":
    main()
