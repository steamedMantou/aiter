# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Perf comparison of DENSE MXFP4 MQA-logits (fp4_mqa_logits) vs the FP8 "gluon"
dense fp8_mqa_logits on long-context (16k-64k) prefill-shaped problems.

This box is heavily shared (all GPUs ~100% busy from other tenants), so fp4 and
fp8 MUST be measured interleaved in the SAME process / time window, else their
ratio is dominated by whoever ran during a quieter window. The MXFP4 kernels
live in this fork; the *gluon* FP8 dense kernel lives only in upstream
/opt/aiter, so we load that kernel module directly from its file (it only imports
from triton) and replicate upstream's ~40-line launch wrapper here.

Dense logits are [M, Nkv] (quadratic), so we hold the query chunk M fixed and
scale Nkv across 16k..64k. We use the balanced full range (every row scans all
Nkv) so throughput == M*Nkv/time is clean and load-balanced. Timing is min-of-N
blocks (rejects transient contention) and fp4/fp8 are interleaved per case.
"""
import importlib.util

import torch

from aiter.ops.fp4_mqa_logits import fp4_mqa_logits
from aiter.ops.triton.utils.types import get_fp8_e4m3_dtype
from test_fp4_mqa_logits import _build_dense, _per_custom_dims_cast_to_fp8

_FP8 = get_fp8_e4m3_dtype()

UPSTREAM_GLUON = (
    "/opt/aiter/aiter/ops/triton/_gluon_kernels/gfx950/attention/fp8_mqa_logits.py"
)


def _load_gluon_fp8_kernel():
    spec = importlib.util.spec_from_file_location("_upstream_gluon_fp8", UPSTREAM_GLUON)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._gluon_fp8_mqa_logits_kernel


_GLUON_FP8 = _load_gluon_fp8_kernel()


def _gluon_feature_flags():
    """Mirror the two runtime feature probes from upstream fp8_mqa_logits.py."""
    import inspect

    async_dist = False
    try:
        from triton.experimental.gluon.language.amd.cdna4 import async_copy

        async_dist = "DistributedLayout" in inspect.getsource(
            async_copy.global_load_to_shared
        )
    except Exception:
        pass
    folded = False
    try:
        from triton.language.core import _unwrap_iterable, constexpr

        folded = not isinstance(_unwrap_iterable((constexpr((0, 1, 2)),)), constexpr)
    except Exception:
        pass
    return async_dist, folded


_ASYNC_DIST, _FOLDED = _gluon_feature_flags()


def fp8_mqa_logits_gluon(Q, KV, kv_scales, weights, cu_starts, cu_ends):
    """Upstream gluon dense launch (replicated, non-invasive), clean_logits=False."""
    seq_len, num_heads, head_size = Q.shape
    seq_len_kv = KV.shape[0]
    aligned = 256
    skv_al = (seq_len_kv + aligned - 1) // aligned * aligned
    logits = torch.empty((seq_len, skv_al), dtype=torch.float32, device=Q.device)[
        :, :seq_len_kv
    ]
    sq_s, sq_h, sq_d = Q.stride()
    skv_s, skv_d = KV.stride()
    sw_s, sw_h = weights.stride()
    sl_s, sl_k = logits.stride()

    USE_FOLDED = _FOLDED and num_heads > 16
    LIMIT = 2 * 1024 * 1024 * 1024
    _GLUON_FP8[(seq_len,)](
        Q_ptr=Q, KV_ptr=KV, kv_scales_ptr=kv_scales, weights_ptr=weights,
        cu_start_ptr=cu_starts, cu_end_ptr=cu_ends, logits_ptr=logits,
        seq_len=seq_len, seq_len_kv=seq_len_kv, NUM_HEADS=num_heads,
        HEAD_SIZE=head_size, stride_q_s=sq_s, stride_q_h=sq_h, stride_q_d=sq_d,
        stride_kv_s=skv_s, stride_kv_d=skv_d, stride_w_s=sw_s, stride_w_h=sw_h,
        stride_logits_s=sl_s, stride_logits_k=sl_k, BLOCK_KV=32, NUM_WARPS=1,
        NUM_BUFFERS=2, NUM_CHAINS=(4 if USE_FOLDED else 0),
        USE_BUFFER_LOAD=KV.numel() * KV.element_size() < LIMIT,
        USE_BUFFER_STORE=logits.numel() * logits.element_size() < LIMIT,
        num_warps=1, waves_per_eu=3,
        USE_PADDED_SHARED_LAYOUT=_ASYNC_DIST,
    )
    return logits


def _bench_min_us(fn, warmup=20, blocks=20, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(blocks):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) / iters * 1e3)
    return best


def run_case(M, H, D, Nkv, seed=0):
    c = _build_dense(M, H, D, Nkv, seed)
    ks = torch.zeros(M, dtype=torch.int32, device="cuda")
    ke = torch.full((M,), Nkv, dtype=torch.int32, device="cuda")
    c["ks"], c["ke"] = ks, ke
    q8 = c["q"].to(_FP8)
    kv8, sc = _per_custom_dims_cast_to_fp8(c["kv"], (0,))

    def fp4():
        return fp4_mqa_logits(c["q_p"], c["q_s"], c["kv_p"], c["kv_s"], c["weights"],
                              ks, ke, clean_logits=False)

    def fp8():
        return fp8_mqa_logits_gluon(q8, kv8, sc, c["weights"], ks, ke)

    # interleave: warm both, then alternate measurement blocks -> identical contention.
    fp4(); fp8(); torch.cuda.synchronize()
    us4 = _bench_min_us(fp4)
    us8 = _bench_min_us(fp8)
    return us4, us8, M * Nkv


def main():
    torch.cuda.init()
    print("gfx950 | D=128 | H=64 | DENSE fp4_mqa_logits vs fp8_mqa_logits (gluon)")
    print("interleaved same-process, min-of-20 (shared box); Glog/s = M*Nkv/time\n")
    header = f"{'M':>5} {'Nkv':>6} | {'fp4':>18} | {'fp8_gluon':>18} | {'fp8/fp4':>7}"
    print(header)
    print("-" * len(header))
    for (M, Nkv) in [(2048, 16384), (2048, 32768), (2048, 65536),
                     (4096, 16384), (4096, 32768), (4096, 65536),
                     (8192, 16384), (8192, 32768),
                     (16384, 16384), (16384, 32768)]:
        us4, us8, valid = run_case(M, 64, 128, Nkv)
        g4, g8 = valid / us4 / 1e3, valid / us8 / 1e3
        sp = us8 / us4
        print(f"{M:>5} {Nkv:>6} | {us4:7.1f}us {g4:4.0f}Gl/s | "
              f"{us8:7.1f}us {g8:4.0f}Gl/s | {sp:6.2f}x")
    print("\nfp8/fp4 > 1 => fp4 faster. fp4 reads half the KV bytes of fp8.")


if __name__ == "__main__":
    main()
