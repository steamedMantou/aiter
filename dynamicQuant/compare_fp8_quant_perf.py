import argparse
import statistics

import torch
import triton

from block_quant import (
    _quantize_fp8_e8m0_vec_kernel,
    block_quantize,
    quantize_fp8_e8m0_triton,
)


def make_input(heads: int, seq_len: int, head_dim: int, dtype: torch.dtype) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn((heads, seq_len, head_dim), device="cuda", dtype=dtype).contiguous()


def bench_cuda(fn, warmup: int, iters: int):
    out = None
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()

    times_ms = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))

    return out, {
        "median_ms": statistics.median(times_ms),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
    }


def quantize_fp8_e8m0_vec_triton(x: torch.Tensor, group_size: int = 128):
    """Force the original one-row-per-program E8M0 FP8 kernel."""
    assert x.shape[-1] % group_size == 0
    assert x.is_cuda

    orig_shape = x.shape
    n_cols = orig_shape[-1]
    num_rows = x.numel() // n_cols
    x_flat = x.reshape(num_rows, n_cols).contiguous()
    num_groups = n_cols // group_size

    out_fp8 = torch.empty(num_rows, n_cols, dtype=torch.float8_e4m3fn, device=x.device)
    out_scale = torch.empty(num_rows, num_groups, dtype=torch.uint8, device=x.device)

    block_groups = min(num_groups, 16)
    _quantize_fp8_e8m0_vec_kernel[(num_rows,)](
        x_flat,
        out_fp8,
        out_scale,
        n_cols,
        N=n_cols,
        GROUP_SIZE=group_size,
        NUM_GROUPS=num_groups,
        FP8_MAX=448.0,
        BLOCK_GROUPS=block_groups,
    )

    return out_fp8.reshape(orig_shape), out_scale.reshape(orig_shape[:-1] + (num_groups,))


def dequant_e8m0_fp8(
    q: torch.Tensor, scale_byte: torch.Tensor, group_size: int
) -> torch.Tensor:
    scale = torch.pow(
        torch.tensor(2.0, device=q.device, dtype=torch.float32),
        scale_byte.to(torch.float32) - 127.0,
    )
    scale = scale.repeat_interleave(group_size, dim=-1)
    return q.to(torch.float32) * scale


def dequant_block_fp8(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.to(torch.float32) * scale[:, :, None]


def error_stats(name: str, dq: torch.Tensor, ref: torch.Tensor):
    ref = ref.to(torch.float32)
    diff = dq - ref
    print(f"\n{name} error vs fp32 input")
    print(f"  max_abs  : {diff.abs().max().item():.6e}")
    print(f"  mean_abs : {diff.abs().mean().item():.6e}")
    print(f"  rmse     : {torch.sqrt(torch.mean(diff * diff)).item():.6e}")
    print(
        "  cosine   : "
        f"{torch.nn.functional.cosine_similarity(dq.flatten(), ref.flatten(), dim=0).item():.8f}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compare FP8 quantization performance across rowblock, vec, and block kernels."
    )
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument(
        "--perf-only",
        action="store_true",
        help="Skip fp32 dequant/error checks to support very large sequence lengths.",
    )
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA is required"
    assert args.head_dim == 128, "This script uses head_dim=128 for block_quantize baseline"
    assert args.head_dim % args.group_size == 0, "head_dim must be divisible by group_size"

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    x = make_input(args.heads, args.seq_len, args.head_dim, dtype)

    seqlen_csum_cu = torch.tensor([0, args.seq_len], dtype=torch.int32, device=x.device)
    seqlens_list = [args.seq_len]

    def run_e8m0_rowblock_fp8():
        return quantize_fp8_e8m0_triton(x, group_size=args.group_size)

    def run_e8m0_vec_fp8():
        return quantize_fp8_e8m0_vec_triton(x, group_size=args.group_size)

    def run_block_fp8():
        return block_quantize(
            input=x,
            seqlen_csum_cu=seqlen_csum_cu,
            seqlens_list=seqlens_list,
            granularity=1,
            quant_dtype="fp8",
        )

    print(f"Input shape: {tuple(x.shape)}, dtype={x.dtype}")
    print("Scale granularity:")
    print(f"  _quantize_fp8_e8m0_rowblock_kernel: {args.group_size} values/share one E8M0 scale")
    print(f"  _quantize_fp8_e8m0_vec_kernel     : {args.group_size} values/share one E8M0 scale")
    print(f"  block_quantize_kernel             : {args.head_dim} values/share one fp32 scale")

    (row_q, row_s), row_time = bench_cuda(run_e8m0_rowblock_fp8, args.warmup, args.iters)
    row_info = (row_q.dtype, row_s.dtype, tuple(row_s.shape))

    if args.perf_only:
        del row_q, row_s
        torch.cuda.empty_cache()

    (vec_q, vec_s), vec_time = bench_cuda(run_e8m0_vec_fp8, args.warmup, args.iters)
    vec_info = (vec_q.dtype, vec_s.dtype, tuple(vec_s.shape))

    if args.perf_only:
        del vec_q, vec_s
        torch.cuda.empty_cache()

    (block_q, block_s, _), block_time = bench_cuda(run_block_fp8, args.warmup, args.iters)
    block_info = (block_q.dtype, block_s.dtype, tuple(block_s.shape))

    print("\nOutput check")
    print(f"  rowblock e8m0 fp8 dtype: {row_info[0]}, scale dtype={row_info[1]}, scale shape={row_info[2]}")
    print(f"  vec e8m0 fp8 dtype     : {vec_info[0]}, scale dtype={vec_info[1]}, scale shape={vec_info[2]}")
    print(f"  block fp8 dtype        : {block_info[0]}, scale dtype={block_info[1]}, scale shape={block_info[2]}")

    if not args.perf_only:
        row_dq = dequant_e8m0_fp8(row_q, row_s, args.group_size)
        vec_dq = dequant_e8m0_fp8(vec_q, vec_s, args.group_size)
        block_dq = dequant_block_fp8(block_q, block_s)
        error_stats("Rowblock E8M0 FP8", row_dq, x)
        error_stats("Vec E8M0 FP8", vec_dq, x)
        error_stats("Block FP8", block_dq, x)

        row_vec = row_dq - vec_dq
        row_block = row_dq - block_dq
        print("\nDequant output diff: Rowblock E8M0 FP8 vs Vec E8M0 FP8")
        print(f"  max_abs  : {row_vec.abs().max().item():.6e}")
        print(f"  mean_abs : {row_vec.abs().mean().item():.6e}")
        print("\nDequant output diff: Rowblock E8M0 FP8 vs Block FP8")
        print(f"  max_abs  : {row_block.abs().max().item():.6e}")
        print(f"  mean_abs : {row_block.abs().mean().item():.6e}")
    else:
        print("\nError check skipped (--perf-only)")

    print("\nPerformance")
    print(
        "  _quantize_fp8_e8m0_rowblock_kernel: "
        f"median={row_time['median_ms']:.4f} ms, "
        f"min={row_time['min_ms']:.4f} ms, max={row_time['max_ms']:.4f} ms"
    )
    print(
        "  _quantize_fp8_e8m0_vec_kernel     : "
        f"median={vec_time['median_ms']:.4f} ms, "
        f"min={vec_time['min_ms']:.4f} ms, max={vec_time['max_ms']:.4f} ms"
    )
    print(
        "  block_quantize_kernel             : "
        f"median={block_time['median_ms']:.4f} ms, "
        f"min={block_time['min_ms']:.4f} ms, max={block_time['max_ms']:.4f} ms"
    )
    print(f"  speedup rowblock/block            : {block_time['median_ms'] / row_time['median_ms']:.3f}x")
    print(f"  speedup rowblock/vec              : {vec_time['median_ms'] / row_time['median_ms']:.3f}x")


if __name__ == "__main__":
    main()
