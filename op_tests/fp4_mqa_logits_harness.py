# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Standalone correctness + performance harness for the MXFP4 MQA-logits HIP
kernel (``csrc/kernels/fp4_mqa_logits_kernels.cu``, JIT module
``module_fp4_mqa_logits``).

This harness is FP4-only: unlike ``op_tests/test_fp4_mqa_logits.py`` it does
*not* call the Triton FP8 baselines (whose signatures have drifted from this
branch), so it provides a stable optimization contract for kernel tuning.

Correctness target: each FP4 kernel path must reproduce its *exact* FP4 dequant
reference (calc_diff < 1e-3). Performance metric: device-time latency in
microseconds via CUDA-graph replay (lower is better).

Usage:
    python op_tests/fp4_mqa_logits_harness.py --correctness
    python op_tests/fp4_mqa_logits_harness.py --benchmark
    python op_tests/fp4_mqa_logits_harness.py --profile      # one shape, light
    python op_tests/fp4_mqa_logits_harness.py --correctness --benchmark
"""
import argparse
import os
import sys

# Ensure the repo root is importable so ``aiter`` and ``op_tests`` resolve when
# this file is run directly as a script (sys.path[0] would otherwise be op_tests/).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch

from aiter.ops.fp4_mqa_logits import (
    fp4_mqa_logits,
    fp4_paged_mqa_logits,
    make_fp4_kv_cache,
)

# Reuse the quant / reference / problem builders from the test module. These
# helpers do not touch the FP8 baselines at import time.
from op_tests.test_fp4_mqa_logits import (
    _build_dense,
    _masked_dense,
    _ref_dense,
    bench_us,
    calc_diff,
    compare_paged_blocked,
    make_paged_problem,
    masked_diff_paged,
    ref_paged,
)

# (M, H, D, Nkv) dense problems and (B, N, H, D, kv_len) paged problems used for
# both correctness and timing. D is fixed at 128 (kernel contracts full D=128).
DENSE_SHAPES = [
    (128, 64, 128, 2048),
    (256, 64, 128, 2048),
    (64, 64, 128, 4096),
    (333, 64, 128, 1560),
    (16, 64, 128, 8192),
    (2048, 64, 128, 8192),
]

PAGED_SHAPES = [
    (1, 1, 64, 128, 2048),
    (2, 2, 64, 128, 4096),
    (2, 1, 64, 128, 8192),
    (4, 1, 64, 128, 16384),
    (8, 1, 64, 128, 32768),
]

# (B, N, H, D, kv_len, block_size, preshuffle) blocked-paged correctness cases.
BLOCKED_SHAPES = [
    (1, 1, 64, 128, 2048, 16, False),
    (2, 2, 64, 128, 4096, 32, False),
    (2, 1, 64, 128, 8192, 16, True),
    (2, 2, 64, 128, 4096, 32, True),
]

CORRECTNESS_TOL = 1e-3


# ---------------------------------------------------------------------------
# Dense
# ---------------------------------------------------------------------------
@torch.inference_mode()
def _dense_case(M, H, D, Nkv, seed=0):
    c = _build_dense(M, H, D, Nkv, seed)
    out_fp4 = fp4_mqa_logits(
        c["q_p"], c["q_s"], c["kv_p"], c["kv_s"], c["weights"], c["ks"], c["ke"]
    )
    torch.cuda.synchronize()
    ref_fp4 = _ref_dense(c["q_deq"], c["kv_deq"], c["weights"], c["ks"], c["ke"])
    af4, rf4 = _masked_dense(out_fp4, ref_fp4, c["ks"], c["ke"], Nkv)
    d_fp4_fp4 = calc_diff(af4, rf4)

    n = torch.arange(Nkv, device=out_fp4.device)
    valid = (n[None, :] >= c["ks"][:, None]) & (n[None, :] < c["ke"][:, None])
    mask_ok = torch.equal(torch.isneginf(out_fp4), ~valid)

    def call():
        return fp4_mqa_logits(
            c["q_p"], c["q_s"], c["kv_p"], c["kv_s"], c["weights"], c["ks"], c["ke"]
        )

    return d_fp4_fp4, mask_ok, call


# ---------------------------------------------------------------------------
# Paged (block_size == 1 interleaved)
# ---------------------------------------------------------------------------
@torch.inference_mode()
def _paged_case(B, N, H, D, kv_len, seed=0):
    p = make_paged_problem(B, N, H, D, kv_len, seed=seed)
    kv_cache = make_fp4_kv_cache(p.kv_p, p.kv_s)
    out_fp4 = torch.full(
        (B * N, p.max_model_len), float("-inf"), device="cuda", dtype=torch.float32
    )

    def call():
        return fp4_paged_mqa_logits(
            p.q_p, p.q_s, kv_cache, p.weights, out_fp4,
            p.context_lens, p.block_tables, p.max_model_len,
        )

    call()
    torch.cuda.synchronize()
    ref_fp4 = ref_paged(p, use_fp4=True)
    d_fp4_fp4 = masked_diff_paged(out_fp4, ref_fp4, p)
    return d_fp4_fp4, call


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
def run_correctness():
    ok = True
    print("== FP4 MQA-logits correctness (kernel vs exact fp4 dequant ref) ==")
    for (M, H, D, Nkv) in DENSE_SHAPES:
        d, mask_ok, _ = _dense_case(M, H, D, Nkv)
        good = (d < CORRECTNESS_TOL) and mask_ok
        ok &= good
        print(f"  dense   M={M:>4} H={H} Nkv={Nkv:>5} | diff={d:.2e} mask_ok={mask_ok} "
              f"{'PASS' if good else 'FAIL'}")
    for (B, N, H, D, L) in PAGED_SHAPES:
        d, _ = _paged_case(B, N, H, D, L)
        good = d < CORRECTNESS_TOL
        ok &= good
        print(f"  paged   B={B} N={N} H={H} kv={L:>5} | diff={d:.2e} "
              f"{'PASS' if good else 'FAIL'}")
    for (B, N, H, D, L, bs, ps) in BLOCKED_SHAPES:
        d = compare_paged_blocked(B, N, H, D, L, bs, ps)
        good = d < CORRECTNESS_TOL
        ok &= good
        tag = "preshuffle" if ps else f"segreg(bs={bs})"
        print(f"  paged   B={B} N={N} H={H} kv={L:>5} {tag:>14} | diff={d:.2e} "
              f"{'PASS' if good else 'FAIL'}")
    print(f"CORRECTNESS: {'PASS' if ok else 'FAIL'}")
    return ok


def _geomean(xs):
    import math
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def run_benchmark(iters=100, warmup=20):
    print("== FP4 MQA-logits benchmark (device latency, lower=better) ==")
    lat = []
    for (M, H, D, Nkv) in DENSE_SHAPES:
        _, _, call = _dense_case(M, H, D, Nkv)
        us = bench_us(call, iters=iters, warmup=warmup)
        lat.append(us)
        print(f"BENCH dense  M={M:>4} H={H} Nkv={Nkv:>5} latency_us={us:.2f}")
    for (B, N, H, D, L) in PAGED_SHAPES:
        _, call = _paged_case(B, N, H, D, L)
        us = bench_us(call, iters=iters, warmup=warmup)
        lat.append(us)
        print(f"BENCH paged  B={B} N={N} H={H} kv={L:>5} latency_us={us:.2f}")
    gm = _geomean(lat)
    mean = sum(lat) / len(lat)
    print(f"MEAN_LATENCY_US={mean:.3f}")
    print(f"GEOMEAN_LATENCY_US={gm:.3f}")
    return gm


def run_profile():
    # Light single-shape run, useful for rocprof / profiling attach.
    M, H, D, Nkv = 2048, 64, 128, 8192
    _, _, call = _dense_case(M, H, D, Nkv)
    for _ in range(50):
        call()
    torch.cuda.synchronize()
    us = bench_us(call, iters=50, warmup=10)
    print(f"PROFILE dense M={M} H={H} Nkv={Nkv} latency_us={us:.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--correctness", action="store_true")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args()

    if not (args.correctness or args.benchmark or args.profile):
        args.correctness = args.benchmark = True

    torch.cuda.init()
    ok = True
    if args.correctness:
        ok = run_correctness()
    if args.benchmark:
        run_benchmark(iters=args.iters, warmup=args.warmup)
    if args.profile:
        run_profile()

    if args.correctness and not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
