# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Plot the DSV4 sparse MLA sweep from bench_dsv4_mla_prod_shapes.py --json."""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STYLE = {
    "vLLM Triton (production)": ("#c44e52", "o"),
    "aiter Triton": ("#dd8452", "s"),
    "dsv4_mla_prefill (ASM)": ("#4c72b0", "D"),
    "opus fp8 paged": ("#55a868", "^"),
}

# From the PCP rocprofv3 trace of one ISL=100k request, on one rank.
PCP_SPARSE_ATTN_MS = 917.4
PCP_DEQUANT_GATHER_MS = 141.2
PCP_TOTAL_MS = 3925.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_json")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    data = json.load(open(args.sweep_json))
    heads = [d["heads"] for d in data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5), gridspec_kw={"width_ratios": [1.3, 1]})

    for impl, (color, marker) in STYLE.items():
        xs, ys = [], []
        for d in data:
            if impl in d["ms"]:
                xs.append(d["heads"])
                ys.append(d["ms"][impl])
        if xs:
            ax1.plot(xs, ys, marker=marker, color=color, label=impl, lw=2)
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(heads)
    ax1.set_xticklabels([f"H={h}\nT={d['tokens']}" for h, d in zip(heads, data)])
    ax1.set_ylabel("ms per call")
    ax1.set_xlabel("heads per rank (total work held constant at 128x1024 head-tokens)")
    ax1.set_title("DSV4 sparse MLA prefill, topk=1024 + SWA=128")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=9)
    ax1.set_ylim(bottom=0)
    ax1.annotate("TP=8 today", xy=(16, 1.77), xytext=(17, 2.55),
                 fontsize=9, arrowprops=dict(arrowstyle="->", lw=1))
    ax1.annotate("PCP=8 today", xy=(128, 1.70), xytext=(60, 2.15),
                 fontsize=9, arrowprops=dict(arrowstyle="->", lw=1))

    pcp = next(d for d in data if d["heads"] == 128)
    speedup = pcp["ms"]["vLLM Triton (production)"] / pcp["ms"]["dsv4_mla_prefill (ASM)"]
    attn_after = PCP_SPARSE_ATTN_MS / speedup
    labels = ["measured\n(vLLM Triton)", "projected\n(ASM MLA)"]
    other = PCP_TOTAL_MS - PCP_SPARSE_ATTN_MS - PCP_DEQUANT_GATHER_MS
    attn = [PCP_SPARSE_ATTN_MS, attn_after]
    deq = [PCP_DEQUANT_GATHER_MS, 0.0]
    rest = [other, other]
    x = np.arange(2)
    ax2.bar(x, rest, 0.5, color="#c8c8c8", label="rest of prefill")
    ax2.bar(x, deq, 0.5, bottom=rest, color="#dd8452", label="KV dequant + gather")
    ax2.bar(x, attn, 0.5, bottom=np.array(rest) + np.array(deq), color="#4c72b0",
            label="sparse MLA attention")
    for i, tot in enumerate([sum(v) for v in zip(rest, deq, attn)]):
        ax2.text(i, tot + 70, f"{tot/1000:.2f}s", ha="center", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("GPU time per rank, one 100k prefill (ms)")
    ax2.set_title(f"PCP=8 projection at {speedup:.1f}x on the attention op")
    ax2.legend(fontsize=9, loc="lower left")
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_ylim(0, PCP_TOTAL_MS * 1.15)

    fig.suptitle("Cherry-picked dsv4_mla_prefill assembly kernel vs Triton, gfx950", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
