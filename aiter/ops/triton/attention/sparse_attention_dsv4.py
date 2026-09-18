# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Forward launchers for DeepSeek-V4 sparse MLA attention."""

import torch
import triton

from aiter.ops.triton._triton_kernels.attention.sparse_attention_dsv4 import (
    _pack_dense_prefix_to_ragged_kernel,
    _sparse_attn_prefill_kernel,
)


def build_ragged_indices_from_dense(
    indices: torch.Tensor,
    lengths: torch.Tensor,
    num_rows: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack valid dense row prefixes into CSR indices and row pointers."""
    indices = indices.reshape(indices.shape[0], -1)
    lengths = lengths.to(device=indices.device, dtype=torch.int32).reshape(-1)
    if lengths.numel() != indices.shape[0]:
        raise ValueError(
            f"expected one length per row, got {lengths.shape} for {indices.shape}"
        )

    max_width = indices.shape[1]
    lengths = lengths.clamp(min=0, max=max_width).contiguous()
    indptr = torch.zeros(indices.shape[0] + 1, dtype=torch.int32, device=indices.device)
    torch.cumsum(lengths, dim=0, out=indptr[1:])

    flat = torch.empty(
        indices.shape[0] * max_width,
        dtype=torch.int32,
        device=indices.device,
    )
    if flat.numel() > 0:
        block_size = 128
        _pack_dense_prefix_to_ragged_kernel[
            (indices.shape[0], triton.cdiv(max_width, block_size))
        ](
            indices,
            lengths,
            indptr,
            flat,
            indices.stride(0),
            int(num_rows),
            max_width,
            BLOCK_SIZE=block_size,
        )
    return flat, indptr


def sparse_mla_fwd_dsv4(
    q: torch.Tensor,
    kv: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    softmax_scale: float,
    attn_sink: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run DSV4 sparse MLA prefill from CSR KV indices."""
    if q.ndim != 3:
        raise ValueError(f"expected q=[T,H,D], got {q.shape}")
    if kv.ndim != 2:
        raise ValueError(f"expected kv=[N,D], got {kv.shape}")
    if q.shape[-1] != kv.shape[-1]:
        raise ValueError(f"q/kv head dim mismatch: {q.shape[-1]} vs {kv.shape[-1]}")

    kv_indices = _as_int32_contiguous_1d(kv_indices)
    kv_indptr = _as_int32_contiguous_1d(kv_indptr)
    if kv_indptr.numel() != q.shape[0] + 1:
        raise ValueError(
            f"expected kv_indptr shape [{q.shape[0] + 1}], got {kv_indptr.shape}"
        )

    has_attn_sink = attn_sink is not None
    if attn_sink is None:
        attn_sink = torch.empty(1, device=q.device, dtype=torch.float32)
    else:
        attn_sink = attn_sink.contiguous()
    out = _get_output_buffer(q, out)

    num_queries, num_heads, head_dim = q.shape
    block_d = triton.next_power_of_2(head_dim)

    def grid(meta):
        return num_queries, triton.cdiv(num_heads, meta["BLOCK_H"])

    _sparse_attn_prefill_kernel[grid](
        q,
        kv,
        kv_indices,
        kv_indptr,
        attn_sink,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        kv.stride(0),
        kv.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        num_heads,
        head_dim,
        kv.shape[0],
        float(softmax_scale),
        HAS_ATTN_SINK=has_attn_sink,
        BLOCK_D=block_d,
    )
    return out


def sparse_mla_fwd_dsv4_dense(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    lengths: torch.Tensor,
    softmax_scale: float,
    attn_sink: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run DSV4 sparse MLA prefill from padded dense indices and lengths."""
    kv_indices, kv_indptr = build_ragged_indices_from_dense(
        indices, lengths, num_rows=kv.shape[0]
    )
    return sparse_mla_fwd_dsv4(
        q=q,
        kv=kv,
        kv_indices=kv_indices,
        kv_indptr=kv_indptr,
        softmax_scale=softmax_scale,
        attn_sink=attn_sink,
        out=out,
    )


def _as_int32_contiguous_1d(x: torch.Tensor) -> torch.Tensor:
    if x.dtype == torch.int32 and x.ndim == 1 and x.is_contiguous():
        return x
    return x.to(torch.int32).contiguous()


def _get_output_buffer(q: torch.Tensor, out: torch.Tensor | None) -> torch.Tensor:
    if out is None:
        return torch.empty_like(q)
    if out.shape != q.shape or out.dtype != q.dtype or out.device != q.device:
        raise ValueError(
            "out must match q shape, dtype, and device: "
            f"q={q.shape}/{q.dtype}/{q.device}, "
            f"out={out.shape}/{out.dtype}/{out.device}"
        )
    return out


__all__ = [
    "build_ragged_indices_from_dense",
    "sparse_mla_fwd_dsv4",
    "sparse_mla_fwd_dsv4_dense",
]
