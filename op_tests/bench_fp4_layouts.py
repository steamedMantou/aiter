# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Perf comparison of the FP4 paged MQA-logits KV-cache layouts vs the FP8
"gluon" deepgemm_fp8_paged_mqa_logits, on a shared problem (gfx950).

Variants (all driven from the same bf16 KV pool / block structure):
  FP4 interleaved   : non-preshuffle, per-token fused cache (KVBlockSize=1)
  FP4 segregated    : non-preshuffle paged cache, block_size>1
  FP4 preshuffle    : FP8-gluon preshuffle layout (shuffle_weight data region)
  FP8 non-preshuffle: deepgemm_fp8_paged_mqa_logits(Preshuffle=False, KVBlockSize=1)
  FP8 preshuffle    : deepgemm_fp8_paged_mqa_logits(Preshuffle=True,  KVBlockSize=bs)

Device-time per call is measured with CUDA-graph replay (launch overhead removed).
"""
import random

import torch

from aiter.ops.fp4_mqa_logits import fp4_paged_mqa_logits, make_fp4_kv_cache
from aiter.ops.shuffle import shuffle_weight
from aiter.ops.triton.attention.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits
from aiter.ops.triton.utils.types import get_fp8_e4m3_dtype

from test_fp4_mqa_logits import (
    bench_us,
    calc_diff,
    dequant_mxfp4_rowwise,
    quant_mxfp4_rowwise,
)

_FP8 = get_fp8_e4m3_dtype()
_FP8_MAX = torch.finfo(_FP8).max


# ---------------------------------------------------------------------------
# Shared problem: block-major KV pool + per-block block tables.
# ---------------------------------------------------------------------------
def build_problem(B, N, H, D, kv_length, block_size, var_ratio=0.5, seed=0):
    torch.manual_seed(seed)
    random.seed(seed)
    dev = "cuda"
    mml = 2 * kv_length
    max_logical_blocks = (mml + block_size - 1) // block_size
    num_blocks = B * max_logical_blocks + 1

    ctx = torch.randint(
        int((1 - var_ratio) * kv_length),
        int((1 + var_ratio) * kv_length) + 1,
        (B,), device=dev,
    ).to(torch.int32)

    q = torch.randn((B, N, H, D), device=dev, dtype=torch.bfloat16)
    kv = torch.randn((num_blocks * block_size, D), device=dev, dtype=torch.bfloat16)
    weights = torch.randn((B * N, H), device=dev, dtype=torch.float32)

    max_blk = (int(ctx.max().item()) + block_size - 1) // block_size
    bt = torch.zeros((B, max_blk), device=dev, dtype=torch.int32)  # per-block
    pool = torch.randperm(num_blocks, device=dev, dtype=torch.int32)
    c = 0
    for i in range(B):
        for j in range((int(ctx[i].item()) + block_size - 1) // block_size):
            bt[i, j] = pool[c % num_blocks]
            c += 1

    # per-token block table (interleaved fp4 / fp8 KVBlockSize=1)
    max_tok = max_blk * block_size
    j = torch.arange(max_tok, device=dev)
    bt_tok = (bt[:, (j // block_size)] * block_size + (j % block_size)).to(torch.int32)

    return dict(
        B=B, N=N, H=H, D=D, bs=block_size, mml=mml, ctx=ctx, q=q, kv=kv,
        weights=weights, bt=bt, bt_tok=bt_tok, num_blocks=num_blocks,
    )


def kv_cache_cast_to_fp8(kv_bd, block_size, D, preshuffle):
    """kv_bd: [num_blocks*block_size, D] bf16 -> [num_blocks, block_size, 1, idx] fp8."""
    nb = kv_bd.shape[0] // block_size
    x = kv_bd.view(nb, block_size, 1, D)
    amax = x.abs().float().amax(dim=3, keepdim=True).clamp(1e-4)
    sf = amax / _FP8_MAX
    xs = (x * (1.0 / sf)).to(_FP8)
    pad = (16 - (block_size * 4) % 16) % 16
    idx = D + 4 + pad
    out = torch.zeros((nb, block_size * idx), device=kv_bd.device, dtype=torch.uint8)
    data = xs.view(nb, block_size, D).view(torch.uint8)
    if preshuffle:
        data = shuffle_weight(data.contiguous(), layout=(16, 16))
    out[:, : block_size * D] = data.reshape(nb, block_size * D)
    out[:, block_size * D : block_size * D + 4 * block_size] = sf.view(
        nb, block_size
    ).view(torch.uint8)
    return out.view(nb, block_size, 1, idx)


@torch.no_grad()
def ref_logits_raw(p, kv_chunk=8192):
    """fp32 reference from the raw bf16 q / KV pool (the cross-dtype 'truth')."""
    B, N, D, bs = p["B"], p["N"], p["D"], p["bs"]
    q_deq = p["q"].float()
    kv_deq = p["kv"].float()
    out = torch.full((B * N, p["mml"]), float("-inf"), device="cuda", dtype=torch.float32)
    ctx_list = p["ctx"].tolist()
    for i in range(B):
        ct = ctx_list[i]
        if ct <= 0:
            continue
        j = torch.arange(ct, device="cuda")
        rows = (p["bt"][i, (j // bs).long()].long() * bs + (j % bs))
        K = kv_deq[rows]
        w = p["weights"][i * N : (i + 1) * N]
        for n in range(N):
            limit = ct - N + n
            row = out[i * N + n]
            for c0 in range(0, ct, kv_chunk):
                c1 = min(ct, c0 + kv_chunk)
                s = (q_deq[i, n][:, None, :] * K[c0:c1][None]).sum(-1)
                s = (torch.relu(s) * w[n][:, None]).sum(0)
                jj = torch.arange(c0, c1, device=s.device)
                row[c0:c1] = torch.where(jj <= limit, s, torch.full_like(s, float("-inf")))
    return out


@torch.no_grad()
def ref_logits(p, use_fp4_kv_p, use_fp4_kv_s, kv_chunk=8192):
    """fp32 reference from the dequantized fp4 KV pool (block-major rows)."""
    B, N, D, bs = p["B"], p["N"], p["D"], p["bs"]
    q_p, q_s = quant_mxfp4_rowwise(p["q"].reshape(-1, D).float())
    q_deq = dequant_mxfp4_rowwise(q_p, q_s).view(B, N, p["H"], D)
    kv_deq = dequant_mxfp4_rowwise(use_fp4_kv_p, use_fp4_kv_s)
    out = torch.full((B * N, p["mml"]), float("-inf"), device="cuda", dtype=torch.float32)
    ctx_list = p["ctx"].tolist()
    for i in range(B):
        ct = ctx_list[i]
        if ct <= 0:
            continue
        j = torch.arange(ct, device="cuda")
        rows = (p["bt"][i, (j // bs).long()].long() * bs + (j % bs))
        K = kv_deq[rows]
        w = p["weights"][i * N : (i + 1) * N]
        for n in range(N):
            limit = ct - N + n
            row = out[i * N + n]
            for c0 in range(0, ct, kv_chunk):
                c1 = min(ct, c0 + kv_chunk)
                s = (q_deq[i, n][:, None, :] * K[c0:c1][None]).sum(-1)
                s = (torch.relu(s) * w[n][:, None]).sum(0)
                jj = torch.arange(c0, c1, device=s.device)
                row[c0:c1] = torch.where(jj <= limit, s, torch.full_like(s, float("-inf")))
    return out


def masked_diff(out, ref, p):
    pos = torch.arange(p["mml"], device="cuda")[None].expand(p["B"] * p["N"], -1)
    ridx = torch.arange(p["B"] * p["N"], device="cuda") // p["N"]
    noff = torch.arange(p["B"] * p["N"], device="cuda") % p["N"]
    lim = (p["ctx"][ridx] - p["N"] + noff)[:, None]
    m = pos <= lim
    return calc_diff(out.masked_fill(~m, 0), ref.masked_fill(~m, 0))


def run_case(B, N, H, D, kv_length, block_size, seed=0):
    p = build_problem(B, N, H, D, kv_length, block_size, seed=seed)
    dev = "cuda"

    # FP4 quant of the shared KV pool (block-major rows).
    kv_p, kv_s = quant_mxfp4_rowwise(p["kv"].float())
    q_p, q_s = quant_mxfp4_rowwise(p["q"].reshape(-1, D).float())
    q_p = q_p.view(B, N, H, D // 2)
    q_s = q_s.view(B, N, H, D // 32)
    ref = ref_logits_raw(p)  # bf16 fp32 truth (same for every variant)
    q_fp8 = p["q"].to(_FP8)

    # One persistent output per variant (kept out of the timed region so we time
    # only the kernel, and so CUDA-graph capture in bench_us actually succeeds).
    out = torch.empty((B * N, p["mml"]), device=dev, dtype=torch.float32)

    kv_il = make_fp4_kv_cache(kv_p, kv_s)
    kv_seg = make_fp4_kv_cache(kv_p, kv_s, block_size=block_size, preshuffle=False)
    kv_ps = make_fp4_kv_cache(kv_p, kv_s, block_size=block_size, preshuffle=True)
    kv8_1 = kv_cache_cast_to_fp8(p["kv"], 1, D, preshuffle=False)
    kv8_ps = kv_cache_cast_to_fp8(p["kv"], block_size, D, preshuffle=True)

    calls = {
        "fp4_interleave": lambda o: fp4_paged_mqa_logits(
            q_p, q_s, kv_il, p["weights"], o, p["ctx"], p["bt_tok"], p["mml"]),
        "fp4_segregated": lambda o: fp4_paged_mqa_logits(
            q_p, q_s, kv_seg, p["weights"], o, p["ctx"], p["bt"], p["mml"],
            kv_block_size=block_size),
        "fp4_preshuffle": lambda o: fp4_paged_mqa_logits(
            q_p, q_s, kv_ps, p["weights"], o, p["ctx"], p["bt"], p["mml"],
            preshuffle=True, kv_block_size=block_size),
        "fp8_nonpreshuf": lambda o: deepgemm_fp8_paged_mqa_logits(
            q_fp8, kv8_1, p["weights"], o, p["ctx"], p["bt_tok"], p["mml"],
            Preshuffle=False, KVBlockSize=1),
        "fp8_preshuffle": lambda o: deepgemm_fp8_paged_mqa_logits(
            q_fp8, kv8_ps, p["weights"], o, p["ctx"], p["bt"], p["mml"],
            Preshuffle=True, KVBlockSize=block_size),
    }

    tokens = int((p["ctx"].float().sum().item()) * N)  # KV tokens streamed
    results = {}
    for name, fn in calls.items():
        try:
            out.fill_(float("-inf"))
            fn(out)
            torch.cuda.synchronize()
            d = masked_diff(out, ref, p)
            us = bench_us(lambda: fn(out))
            kv_bytes = tokens * (68 if name.startswith("fp4") else 132)
            out_bytes = tokens * 4
            gbps = (kv_bytes + out_bytes) / (us * 1e-6) / 1e9
            results[name] = (d, us, gbps)
        except Exception as e:
            results[name] = (repr(e)[:36], None, None)
    return p, results


def main():
    torch.cuda.init()
    cols = ["fp4_interleave", "fp4_segregated", "fp4_preshuffle",
            "fp8_nonpreshuf", "fp8_preshuffle"]
    block_size = 16
    print(f"gfx950 | block_size={block_size} | D=128 | next_n=1 | device-us (CUDA-graph)")
    print("us = device time, GB/s = (KV read + logits write)/time "
          "(fp4 72B/tok, fp8 136B/tok)\n")
    header = (f"{'B':>3} {'kv':>6} | " + " ".join(f"{c:>19}" for c in cols)
              + f" | {'fp4ps vs':>22}")
    print(header)
    print(f"{'':>3} {'':>6} | " + " ".join(f"{'':>19}" for _ in cols)
          + f" | {'fp4il fp8np fp8ps':>22}")
    print("-" * len(header))
    for (B, L) in [(16, 16384), (16, 32768), (16, 65536),
                   (32, 16384), (32, 32768), (32, 65536),
                   (64, 16384), (64, 32768), (128, 16384), (128, 32768)]:
        _, r = run_case(B, 1, 64, 128, L, block_size)
        cells = []
        for c in cols:
            d, us, gbps = r[c]
            cells.append((f"{us:6.1f}us {gbps:5.0f}GB/s" if us else "n/a").rjust(19))
        ps = r["fp4_preshuffle"][1]

        def sp(name):
            o = r[name][1]
            return f"{o / ps:.2f}x" if (o and ps) else "-"

        rel = f"{sp('fp4_interleave'):>6} {sp('fp8_nonpreshuf'):>6} {sp('fp8_preshuffle'):>6}"
        print(f"{B:>3} {L:>6} | " + " ".join(cells) + f" | {rel:>22}")
    print("\n'fp4ps vs' = other_us / fp4_preshuffle_us  (>1 => fp4_preshuffle faster).")
    print("d (accuracy, omitted above): fp4 ~1e-2, fp8 ~1e-3 vs fp32 bf16 truth.")


if __name__ == "__main__":
    main()
