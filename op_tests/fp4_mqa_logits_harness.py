# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""GEAK-conformant correctness + performance harness for the MXFP4 MQA-logits
HIP kernel (``csrc/kernels/fp4_mqa_logits_kernels.cu``, JIT module
``module_fp4_mqa_logits``).

FP4-only by design: unlike ``op_tests/test_fp4_mqa_logits.py`` it never calls
the Triton FP8 baselines (whose signatures have drifted from this branch), so it
is a stable optimization contract.

Worktree routing (critical for GEAK parallel evaluation): ``aiter`` is an
editable install, so this harness pins every aiter path to ``$GEAK_WORK_DIR``
*before* importing aiter:

* ``sys.path`` -> WORK_DIR first        (import the patched aiter package)
* ``AITER_META_DIR`` = WORK_DIR         (compile WORK_DIR/csrc, i.e. the patch)
* ``AITER_JIT_DIR``  = WORK_DIR/.geak_aiter_jit  (per-slot build isolation)

Correctness target: each FP4 kernel path must reproduce its *exact* FP4 dequant
reference (calc_diff < 1e-3). Metric: device latency via CUDA-graph replay,
reported as the geometric-mean millisecond value (lower is better).

Flags (universal GEAK harness contract):
    --correctness       run correctness check; print 'CORRECTNESS: PASS'/OK
    --benchmark         time kernels; print GEAK_RESULT_LATENCY_MS=<float>
    --full-benchmark    benchmark + print GEAK_RESULT_SPEEDUP=<float>
    --profile           single-shape run for an external profiler

Markers emitted on --benchmark / --full-benchmark:
    GEAK_RESULT_METRIC=<ms>  GEAK_RESULT_UNIT=ms  GEAK_RESULT_DIRECTION=lower_is_better
    GEAK_RESULT_LATENCY_MS=<ms>
    per-shape lines:  (dense M=.. H=.. Nkv=..): <ms> ms
"""
import argparse
import math
import os
import sys

# --- aiter worktree routing (must run before `import aiter`) ----------------
WORK_DIR = os.environ.get("GEAK_WORK_DIR") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
# Source routing: compile this worktree's csrc/*.cu (the patched kernel).
os.environ.setdefault("AITER_META_DIR", WORK_DIR)
# Build-output routing: per-worktree JIT dir so parallel slots never collide on
# build/, <module>.so or the ninja lock (WORK_DIR differs per slot).
os.environ.setdefault("AITER_JIT_DIR", os.path.join(WORK_DIR, ".geak_aiter_jit"))
# Import the worktree's aiter (and op_tests helpers) ahead of any install.
if WORK_DIR not in sys.path:
    sys.path.insert(0, WORK_DIR)

import torch  # noqa: E402

from aiter.ops.fp4_mqa_logits import (  # noqa: E402
    fp4_mqa_logits,
    fp4_paged_mqa_logits,
    make_fp4_kv_cache,
)

# Reuse the quant / reference / problem builders. Importing this module does not
# *call* the FP8 baselines (only the test functions do), so it imports cleanly.
from op_tests.test_fp4_mqa_logits import (  # noqa: E402
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

# (M, H, D, Nkv) dense and (B, N, H, D, kv_len) paged problems. D fixed at 128.
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
@torch.inference_mode()
def _dense_case(M, H, D, Nkv, seed=0):
    c = _build_dense(M, H, D, Nkv, seed)
    out_fp4 = fp4_mqa_logits(
        c["q_p"], c["q_s"], c["kv_p"], c["kv_s"], c["weights"], c["ks"], c["ke"]
    )
    torch.cuda.synchronize()
    ref_fp4 = _ref_dense(c["q_deq"], c["kv_deq"], c["weights"], c["ks"], c["ke"])
    af4, rf4 = _masked_dense(out_fp4, ref_fp4, c["ks"], c["ke"], Nkv)
    d = calc_diff(af4, rf4)
    n = torch.arange(Nkv, device=out_fp4.device)
    valid = (n[None, :] >= c["ks"][:, None]) & (n[None, :] < c["ke"][:, None])
    mask_ok = torch.equal(torch.isneginf(out_fp4), ~valid)

    def call():
        return fp4_mqa_logits(
            c["q_p"], c["q_s"], c["kv_p"], c["kv_s"], c["weights"], c["ks"], c["ke"]
        )

    return d, mask_ok, call


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
    d = masked_diff_paged(out_fp4, ref_fp4, p)
    return d, call


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
    if ok:
        print("OK")
    return ok


def _bench_iters():
    env = os.environ.get("GEAK_BENCHMARK_ITERATIONS")
    if env:
        try:
            return max(10, int(env))
        except ValueError:
            pass
    return None


def run_benchmark(iters=100, warmup=20):
    env_iters = _bench_iters()
    if env_iters is not None:
        iters = env_iters
    print("== FP4 MQA-logits benchmark (device latency via CUDA-graph replay) ==")
    lat_ms = []
    for (M, H, D, Nkv) in DENSE_SHAPES:
        _, _, call = _dense_case(M, H, D, Nkv)
        ms = bench_us(call, iters=iters, warmup=warmup) / 1e3
        lat_ms.append(ms)
        # Parseable per-shape line: "(label): <ms> ms".
        print(f"(dense M={M} H={H} Nkv={Nkv}): {ms:.6f} ms")
    for (B, N, H, D, L) in PAGED_SHAPES:
        _, call = _paged_case(B, N, H, D, L)
        ms = bench_us(call, iters=iters, warmup=warmup) / 1e3
        lat_ms.append(ms)
        print(f"(paged B={B} N={N} H={H} kv={L}): {ms:.6f} ms")
    gm = math.exp(sum(math.log(x) for x in lat_ms) / len(lat_ms))
    mean = sum(lat_ms) / len(lat_ms)
    print(f"Geomean (ms): {gm:.6f}")
    print(f"Mean (ms): {mean:.6f}")
    # GEAK scoring markers (geometric-mean latency, lower is better).
    print(f"GEAK_RESULT_METRIC={gm:.6f}")
    print("GEAK_RESULT_UNIT=ms")
    print("GEAK_RESULT_DIRECTION=lower_is_better")
    print(f"GEAK_RESULT_LATENCY_MS={gm:.6f}")
    return gm


def run_full_benchmark(iters=200, warmup=40):
    gm = run_benchmark(iters=iters, warmup=warmup)
    # Speedup vs an optional externally-provided baseline (GEAK also computes
    # this deterministically from benchmark_baseline.txt). Default 1.0 when the
    # baseline metric is not supplied.
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
    M, H, D, Nkv = 2048, 64, 128, 8192
    _, _, call = _dense_case(M, H, D, Nkv)
    for _ in range(50):
        call()
    torch.cuda.synchronize()
    us = bench_us(call, iters=50, warmup=10)
    print(f"PROFILE dense M={M} H={H} Nkv={Nkv} latency_ms={us / 1e3:.6f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--correctness", action="store_true")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--full-benchmark", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--iters", "--iterations", dest="iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
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
