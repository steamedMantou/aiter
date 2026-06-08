// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Torch <-> HIP glue for the MXFP4 MQA-logits kernels. The device kernels and
// their C-ABI launchers live in csrc/kernels/fp4_mqa_logits_kernels.cu; here we
// unwrap the torch tensors, pick the current HIP stream and forward to them.
#include <torch/all.h>
#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include "fp4_mqa_logits.h"

// Align(D/2 + D/32, 16) = align(68, 16) -- the fused paged-cache row stride that
// make_fp4_kv_cache() produces and the paged kernel reads.
#define FP4_KV_STRIDE 80

// C-ABI launchers defined in the vendored kernel TU.
extern "C" {
void launch_fp4_pa_mqa(const uint8_t* q_p,
                       const uint8_t* q_s,
                       const uint8_t* kv,
                       const int* block_tables,
                       const float* weights,
                       float* out,
                       const int* context_lens,
                       int B,
                       int N,
                       int H,
                       int max_model_len,
                       int max_block_len,
                       int kv_stride,
                       int split_kv,
                       long stream);

void launch_fp4_mqa_dense(const uint8_t* q_p,
                          const uint8_t* q_s,
                          const uint8_t* kv_p,
                          const uint8_t* kv_s,
                          const float* weights,
                          const int* cu_starts,
                          const int* cu_ends,
                          float* out,
                          int M,
                          int Nkv,
                          int H,
                          int kv_p_stride,
                          int kv_s_stride,
                          int out_stride,
                          int split_kv,
                          long stream);

void launch_fp4_mqa_dense32(const uint8_t* q_p,
                            const uint8_t* q_s,
                            const uint8_t* kv_p,
                            const uint8_t* kv_s,
                            const float* weights,
                            const int* cu_starts,
                            const int* cu_ends,
                            float* out,
                            int M,
                            int Nkv,
                            int H,
                            int kv_p_stride,
                            int kv_s_stride,
                            int out_stride,
                            int split_kv,
                            long stream);
}

namespace aiter {

static inline long current_hip_stream()
{
    return reinterpret_cast<long>(
        static_cast<hipStream_t>(at::hip::getCurrentHIPStream()));
}

torch::Tensor fp4_mqa_logits(torch::Tensor& q_p,
                             torch::Tensor& q_s,
                             torch::Tensor& kv_p,
                             torch::Tensor& kv_s,
                             torch::Tensor& weights,
                             torch::Tensor& cu_starts,
                             torch::Tensor& cu_ends,
                             torch::Tensor& out,
                             int64_t split_kv,
                             int64_t core)
{
    const int M   = static_cast<int>(weights.size(0));
    const int H   = static_cast<int>(weights.size(1));
    const int Nkv = static_cast<int>(kv_p.size(0));

    auto* launch = (core == 16) ? &launch_fp4_mqa_dense : &launch_fp4_mqa_dense32;
    launch(reinterpret_cast<const uint8_t*>(q_p.data_ptr()),
           reinterpret_cast<const uint8_t*>(q_s.data_ptr()),
           reinterpret_cast<const uint8_t*>(kv_p.data_ptr()),
           reinterpret_cast<const uint8_t*>(kv_s.data_ptr()),
           reinterpret_cast<const float*>(weights.data_ptr()),
           reinterpret_cast<const int*>(cu_starts.data_ptr()),
           reinterpret_cast<const int*>(cu_ends.data_ptr()),
           reinterpret_cast<float*>(out.data_ptr()),
           M,
           Nkv,
           H,
           static_cast<int>(kv_p.stride(0)),
           static_cast<int>(kv_s.stride(0)),
           static_cast<int>(out.stride(0)),
           static_cast<int>(split_kv),
           current_hip_stream());
    return out;
}

torch::Tensor fp4_paged_mqa_logits(torch::Tensor& q_p,
                                   torch::Tensor& q_s,
                                   torch::Tensor& kv_cache,
                                   torch::Tensor& block_tables,
                                   torch::Tensor& weights,
                                   torch::Tensor& context_lens,
                                   torch::Tensor& out_logits,
                                   int64_t max_model_len,
                                   int64_t split_kv)
{
    const int B             = static_cast<int>(q_p.size(0));
    const int N             = static_cast<int>(q_p.size(1));
    const int H             = static_cast<int>(q_p.size(2));
    const int max_block_len = static_cast<int>(block_tables.size(1));

    launch_fp4_pa_mqa(reinterpret_cast<const uint8_t*>(q_p.data_ptr()),
                      reinterpret_cast<const uint8_t*>(q_s.data_ptr()),
                      reinterpret_cast<const uint8_t*>(kv_cache.data_ptr()),
                      reinterpret_cast<const int*>(block_tables.data_ptr()),
                      reinterpret_cast<const float*>(weights.data_ptr()),
                      reinterpret_cast<float*>(out_logits.data_ptr()),
                      reinterpret_cast<const int*>(context_lens.data_ptr()),
                      B,
                      N,
                      H,
                      static_cast<int>(max_model_len),
                      max_block_len,
                      FP4_KV_STRIDE,
                      static_cast<int>(split_kv),
                      current_hip_stream());
    return out_logits;
}

} // namespace aiter
