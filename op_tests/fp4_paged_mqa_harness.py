# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""GEAK harness focused on the PAGED MXFP4 MQA-logits decode kernel
(``fp4_paged_mqa_logits``, preshuffle KV layout -> dedicated
``fp4_pa_mqa_preshuffle_kernel`` in csrc/kernels/fp4_mqa_logits_kernels.cu).

Goal: beat the FP8 "gluon" preshuffle paged kernel on the decode shapes that
currently regress (B in {4,8,16}, kv in {8192,16384}), i.e. drive
``fp4_preshuffle / fp8 < 1`` across the board.

Scoring metric (minimize): geometric-mean device latency (ms) of the FP4 paged
preshuffle kernel over the target shapes. The harness also prints, for human /
agent visibility, the per-shape ratio against a fixed FP8 gluon reference
(measured on this gfx950 box) and a win count -- ratio < 1 means FP4 wins.

Worktree routing (same contract as op_tests/fp4_mqa_logits_harness.py): pin
aiter to $GEAK_WORK_DIR before importing aiter so each GEAK slot compiles its
own patched kernel.

Flags: --correctness --benchmark --full-benchmark --profile
Markers: GEAK_RESULT_LATENCY_MS / GEAK_RESULT_METRIC / GEAK_RESULT_SPEEDUP
"""
import argparse
import math
import os
import sys

WORK_DIR = os.environ.get("GEAK_WORK_DIR") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
os.environ.setdefault("AITER_META_DIR", WORK_DIR)  # compile this worktree's csrc
os.environ.setdefault("AITER_JIT_DIR", os.path.join(WORK_DIR, ".geak_aiter_jit"))
if WORK_DIR not in sys.path:
    sys.path.insert(0, WORK_DIR)

import torch  # noqa: E402

from aiter.ops.fp4_mqa_logits import fp4_paged_mqa_logits, make_fp4_kv_cache  # noqa: E402

from op_tests.test_fp4_mqa_logits import (  # noqa: E402
    bench_us,
    make_blocked_paged_problem,
    masked_diff_blocked,
    ref_blocked_paged,
)

# Target decode shapes (B, kv) with N=1, H=64, D=128. These are the rows of the
# single-op table where fp4_preshuffle still lost to fp8 gluon.
H, D, NEXT_N = 64, 128, 1
BLOCK_SIZE = 64  # preshuffle requires block_size % 16 == 0; dedicated kernel: {16,32,64}
TARGET_SHAPES = [
    (4, 8192), (4, 16384),
    (8, 8192), (8, 16384),
    (16, 8192), (16, 16384),
]

# FP8 gluon preshuffle reference latency (us), measured on this gfx950 box
# (single-op CUDA-graph device-us table). Used only for ratio reporting.
FP8_REF_US = {
    (4, 8192): 10.03, (4, 16384): 9.97,
    (8, 8192): 10.01, (8, 16384): 10.91,
    (16, 8192): 10.95, (16, 16384): 15.77,
}

CORRECTNESS_TOL = 1e-3


@torch.inference_mode()
def _build(B, kv, var_ratio, seed=0):
    p = make_blocked_paged_problem(
        B, NEXT_N, H, D, kv, BLOCK_SIZE, var_ratio=var_ratio, seed=seed
    )
    kv_cache = make_fp4_kv_cache(p.kv_p, p.kv_s, block_size=BLOCK_SIZE, preshuffle=True)
    out = torch.full(
        (B * NEXT_N, p.max_model_len), float("-inf"), device="cuda", dtype=torch.float32
    )

    def call():
        return fp4_paged_mqa_logits(
            p.q_p, p.q_s, kv_cache, p.weights, out, p.context_lens, p.block_tables,
            p.max_model_len, preshuffle=True, kv_block_size=BLOCK_SIZE,
        )

    return p, out, call


def run_correctness():
    ok = True
    print("== FP4 paged preshuffle correctness (kernel vs exact fp4 dequant ref) ==")
    for (B, kv) in TARGET_SHAPES:
        p, out, call = _build(B, kv, var_ratio=0.5)
        call()
        torch.cuda.synchronize()
        ref = ref_blocked_paged(p)
        d = masked_diff_blocked(out, ref, p)
        good = d < CORRECTNESS_TOL
        ok &= good
        print(f"  preshuf B={B:>2} N={NEXT_N} H={H} kv={kv:>5} bs={BLOCK_SIZE} | "
              f"diff={d:.2e} {'PASS' if good else 'FAIL'}")
    print(f"CORRECTNESS: {'PASS' if ok else 'FAIL'}")
    if ok:
        print("OK")
    return ok


def _bench_one(B, kv, iters, warmup):
    # Pin kv (var_ratio=0) so the measured shape matches the table semantics.
    _, _, call = _build(B, kv, var_ratio=0.0)
    return bench_us(call, iters=iters, warmup=warmup)


def run_benchmark(iters=200, warmup=40):
    env = os.environ.get("GEAK_BENCHMARK_ITERATIONS")
    if env:
        try:
            iters = max(20, int(env))
        except ValueError:
            pass
    print("== FP4 paged preshuffle benchmark (device latency, vs fp8 gluon ref) ==")
    lat_ms, ratios, wins = [], [], 0
    for (B, kv) in TARGET_SHAPES:
        us = _bench_one(B, kv, iters, warmup)
        ms = us / 1e3
        lat_ms.append(ms)
        ref = FP8_REF_US.get((B, kv))
        ratio = (us / ref) if ref else float("nan")
        if ref and us < ref:
            wins += 1
        ratios.append(ratio)
        print(f"(paged B={B} N={NEXT_N} H={H} kv={kv} bs={BLOCK_SIZE}): {ms:.6f} ms  "
              f"[fp4={us:6.2f}us fp8={ref:6.2f}us  fp4/fp8={ratio:.3f}"
              f"{'  WIN' if ref and us < ref else ''}]")
    gm = math.exp(sum(math.log(x) for x in lat_ms) / len(lat_ms))
    gm_ratio = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
    print(f"Geomean (ms): {gm:.6f}")
    print(f"Geomean fp4/fp8 ratio: {gm_ratio:.4f}  (<1 = fp4 faster than fp8 gluon)")
    print(f"WINS vs fp8: {wins}/{len(TARGET_SHAPES)}")
    print(f"GEAK_RESULT_METRIC={gm:.6f}")
    print("GEAK_RESULT_UNIT=ms")
    print("GEAK_RESULT_DIRECTION=lower_is_better")
    print(f"GEAK_RESULT_LATENCY_MS={gm:.6f}")
    return gm


def run_full_benchmark(iters=300, warmup=60):
    gm = run_benchmark(iters=iters, warmup=warmup)
    base = os.environ.get("GEAK_BASELINE_METRIC_MS") or os.environ.get(
        "GEAK_BASELINE_LATENCY_MS"
    )
    speedup = 1.0
    if base:
        try:
            b = float(base)
            if b > 0 and gm > 0:
                speedup = b / gm
        except ValueError:
            pass
    print(f"GEAK_RESULT_SPEEDUP={speedup:.6f}")
    return gm


def run_profile():
    B, kv = 16, 16384
    _, _, call = _build(B, kv, var_ratio=0.0)
    for _ in range(50):
        call()
    torch.cuda.synchronize()
    us = bench_us(call, iters=50, warmup=10)
    print(f"PROFILE paged B={B} kv={kv} latency_ms={us / 1e3:.6f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--correctness", action="store_true")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--full-benchmark", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--iters", "--iterations", dest="iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=40)
    args = ap.parse_args()

    if not (args.correctness or args.benchmark or args.full_benchmark or args.profile):
        args.correctness = args.benchmark = True

    torch.cuda.init()
    ok = True
    if args.correctness:
        ok = run_correctness()
    if args.full_benchmark:
        run_full_benchmark()
    elif args.benchmark:
        run_benchmark(iters=args.iters, warmup=args.warmup)
    if args.profile:
        run_profile()

    if args.correctness and not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
