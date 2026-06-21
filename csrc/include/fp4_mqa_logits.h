#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// MXFP4 MQA-logits ops -- the FP4 counterparts of the FP8 `fp8_mqa_logits`
// (dense / ragged prefill) and `fp8_paged_mqa_logits` (paged decode). Q and K
// are MXFP4 (packed e2m1 data + e8m0 micro-scale), so unlike the FP8 ops there
// is no separate per-row kv_scales tensor -- the scales are the q_s / kv_s
// exponent planes. Backed by the hand-written gfx950 (CDNA4) kernels in
// csrc/kernels/fp4_mqa_logits_kernels.cu.
#include <torch/extension.h>

namespace aiter {

// Dense (non-paged) MXFP4 MQA-logits.
//   logits[m, n] = sum_h relu(Q[m,h,:] . K[n,:]) * weights[m,h]
//   for cu_starts[m] <= n < cu_ends[m], else left untouched (caller pre-fills).
// q_p [M,H,D/2] u8, q_s [M,H,D/32] u8, kv_p [Nkv,D/2] u8, kv_s [Nkv,D/32] u8,
// weights [M,H] f32, cu_starts/cu_ends [M] i32, out [M,Nkv] f32 (written here).
// core: 32 -> 32x32x64 matrix core (default), 16 -> 16x16x128 core.
torch::Tensor fp4_mqa_logits(torch::Tensor& q_p,
                             torch::Tensor& q_s,
                             torch::Tensor& kv_p,
                             torch::Tensor& kv_s,
                             torch::Tensor& weights,
                             torch::Tensor& cu_starts,
                             torch::Tensor& cu_ends,
                             torch::Tensor& out,
                             int64_t split_kv,
                             int64_t core);

// Paged MXFP4 MQA-logits. Two KV-cache layouts, matching the FP8 "gluon"
// deepgemm_fp8_paged_mqa_logits input layout (selected by `layout`):
//   layout 0 (interleaved, non-preshuffle, KVBlockSize==1): per-token fused 80B
//     rows, block_tables[b,j] is a per-token slot, tok(j) = block_tables[b,j].
//     kv_cache [num_blocks, 80] u8 (fused e2m1|e8m0|pad).
//   layout 1 (segregated, non-preshuffle, KVBlockSize>1): paged cache, per
//     physical block [block_size*D/2 e2m1 | block_size*D/32 e8m0 | pad];
//     block_tables[b, j/block_size] maps logical->physical, blk_stride = bytes
//     per physical block (= kv_cache.stride(0), page padding included).
//   layout 2 (preshuffle, KVBlockSize multiple of 16): the FP8-gluon preshuffle
//     layout; same per-block block_tables/blk_stride, data region swizzled.
//   logits[b*N+n, j] = sum_h relu(Q[b,n,h,:] . K[tok(j),:]) * w[b*N+n,h]
//   for j <= context_lens[b] - N + n, else left untouched.
// q_p [B,N,H,D/2] u8, q_s [B,N,H,D/32] u8, block_tables [B, max_block_len] i32,
// weights [B*N,H] f32, context_lens [B] i32, out_logits [B*N, max_model_len] f32.
torch::Tensor fp4_paged_mqa_logits(torch::Tensor& q_p,
                                   torch::Tensor& q_s,
                                   torch::Tensor& kv_cache,
                                   torch::Tensor& block_tables,
                                   torch::Tensor& weights,
                                   torch::Tensor& context_lens,
                                   torch::Tensor& out_logits,
                                   int64_t max_model_len,
                                   int64_t split_kv,
                                   int64_t layout,
                                   int64_t block_size,
                                   int64_t blk_stride);

} // namespace aiter
