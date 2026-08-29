// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Prebuilt-code-object entry points for the DeepSeek-V4 MLA sparse prefill
// kernels (the "h40" fp8 prefill family).
//
// The device code is compiled ahead of time into
// hsa/gfx950/dsv4_mla_prefill/dsv4_mla_prefill.co from the kernel headers under
// csrc/include, so nothing here JIT-compiles a kernel: only this loader is built,
// and the attention kernel's schedule is fixed at ship time rather than left to
// whatever clang the deployment image happens to carry.  That matters for this
// kernel in particular -- it sits at 504 of 512 ArchVGPRs with zero scratch, so
// a compiler that costs it eight registers spills the inner loop.
//
// What stays here is the half a code object cannot hold: argument validation,
// the layout descriptor, and the launch geometry.  Those bodies are carried
// over verbatim from the JIT launcher (csrc/py_itfs_cu/
// pa_sparse_prefill_opus_kernels.cu); only the <<<>>> became launch_kernel.
//
// How the shipped object was built, so a rebuild can be compared against it:
//
//   hipcc -O3 -std=c++20 --offload-arch=gfx950 -I csrc/include
//         -DWITH_HIP -D_GLIBCXX_USE_CXX11_ABI=1 -DENABLE_CK=1
//         -DENABLE_ROPE_POSITIONS_INT32=0 -D__HIP_PLATFORM_AMD__=1
//         -D__HIP_PLATFORM_HCC__=1 -DUSE_ROCM=1 -DHIPBLAS_V2 -DCUDA_HAS_FP16=1
//         -DLEGACY_HIPBLAS_DIRECT -DUSE_PROF_API=1
//         -ffast-math -fgpu-flush-denormals-to-zero -fno-offload-uniform-block
//         -fno-gpu-rdc -mcmodel=large -fno-unique-section-names
//         -ffunction-sections -fdata-sections -fvisibility=hidden
//         -fvisibility-inlines-hidden -fPIC
//         -mllvm --amdgpu-kernarg-preload-count=32
//         -mllvm --lsr-drop-solution=1
//         -mllvm -amdgpu-early-inline-all=false
//         -mllvm -amdgpu-function-calls=false
//         -mllvm -enable-post-misched=1
//         -c -x hip <TU including pa_sparse_prefill_fp8_h40.h>
//
// then llvm-objcopy --dump-section=.hip_fatbin and clang-offload-bundler
// --unbundle --targets=hipv4-amdgcn-amd-amdhsa--gfx950: hipModuleLoad wants a
// bare ELF, and a bundled object builds fine and fails at load time.
//
// These are aiter's own JIT flags for module_pa_sparse_prefill_opus, minus the
// torch include paths a device TU does not use.  Two are load-bearing and
// neither is obvious, because both are read wrong from the obvious place:
//
//   -enable-post-misched=1        the generated build.ninja carries =0 and then
//                                 =1.  The later wins; the kernel is 5-9%
//                                 slower with the =0 a casual reading gives.
//   -amdgpu-early-inline-all=false  the ninja *rule* appends this after
//                                 cuda_cflags' =true.  Reading cuda_cflags
//                                 alone gets =true and a different register
//                                 allocation.
//
// Verify a rebuild rather than trusting it: llvm-objdump -d --mcpu=gfx950, and
// both kernels must match instruction for instruction.  Compare the
// disassembly, not the hash -- two builds of the same source at different paths
// differ in .dynstr, because __hip_cuid_ follows the path.

#include "aiter_hip_common.h"
#include "aiter_tensor.h"
#include "aiter_ctypes_error.h"

#include <cstddef>
#include <exception>

// Mirrors PA_FP8_H40_MIN_H in the kernel header: the code object was built
// with that threshold, and this loader does not include the header.
#define PA_FP8_MIN_H 16

namespace {

// ---------------------------------------------------------------------------
// Mirrors pa_fp8_h40::pa_fp8_kargs.  The loader deliberately does not include
// the kernel header: the whole point of shipping a code object is that the
// device source is not compiled here.  The static_asserts are the guard -- the
// .co's kernarg_segment_size is 208, and a field reordered on either side would
// otherwise be a silent wrong answer instead of a build failure.
// ---------------------------------------------------------------------------
struct PaFp8Kargs
{
    const void* q_ptr;
    const void* q_rope_ptr;
    const void* unified_kv_ptr;
    const void* unified_kv_rope_ptr;
    const void* kv_ptr;
    const void* kv_rope_ptr;
    const float* attn_sink_ptr;
    void* out_ptr;
    const int* kv_indptr_prefix;
    const int* kv_indices_prefix;
    const int* kv_indptr_extend;
    const int* kv_indices_extend;
    int N;
    int H;
    int D;
    int total_pages;
    int total_tokens;
    int stride_q_n;
    int stride_q_h;
    int stride_o_n;
    int stride_o_h;
    int stride_kv_row;
    int stride_qr_h;
    int stride_kvr_row;
    // Field names are pa_fp8_kargs' own, deliberately: keeping them identical
    // is what makes the field-by-field check against that struct a glance.
    int sgl_page_shift[2];
    int sgl_rows_per_page[2];
    int sgl_scale_off[2];
    const int* kv_lens_prefix;
    const int* kv_lens_extend;
    int kv_stride_q_prefix;
    int kv_stride_q_extend;
    float softmax_scale;
    const int* max_e_ptr;
};
static_assert(sizeof(PaFp8Kargs) == 208, "kargs layout drifted from the .co");
static_assert(offsetof(PaFp8Kargs, out_ptr) == 56, "kargs layout drifted");
static_assert(offsetof(PaFp8Kargs, N) == 96, "kargs layout drifted");
static_assert(offsetof(PaFp8Kargs, sgl_page_shift) == 144, "kargs layout drifted");
static_assert(offsetof(PaFp8Kargs, softmax_scale) == 192, "kargs layout drifted");
static_assert(offsetof(PaFp8Kargs, max_e_ptr) == 200, "kargs layout drifted");

// The Q pack kernel takes loose parameters rather than one struct, so its
// argument buffer is spelled out here the way HIP packs a kernarg segment:
// natural alignment, no packing attribute.

struct QPackArgs
{
    const void* src;      // const __bf16*
    unsigned char* dst;
    int rows;
    int src_stride;
};
static_assert(sizeof(QPackArgs) == 24 && offsetof(QPackArgs, rows) == 16, "");

constexpr int PA_FP8_H40_D_NOPE_PADDED = 512;  // 448 NoPE fp8 + 14 E8M0 + pad
constexpr int PA_FP8_H40_D_ROPE        = 64;
constexpr int PA_FP8_H40_D_HEAD        = 512;

// pa_fp8_traits, fixed at the moment the code object was built.
constexpr int HEADS_PER_BLOCK = 128;  // NUM_WARPS 4 * Q_SUB 2 * Q_TILE 16
constexpr int BLOCK_SIZE      = 256;  // NUM_WARPS 4 * WARP_SIZE 64

constexpr int ceil_div(int a, int b) { return (a + b - 1) / b; }

// C++ mangled names, taken from `llvm-nm dsv4_mla_prefill.co`.  There is
// exactly one code object and the namespace is unique to it, so a symbol names
// exactly one kernel.
constexpr const char* CO_PATH = "/dsv4_mla_prefill/dsv4_mla_prefill.co";
constexpr const char* SYM_PREFILL  = "_ZN10pa_fp8_h4021pa_prefill_fp8_kernelENS_12pa_fp8_kargsE";
constexpr const char* SYM_Q_PACK   = "_ZN10pa_fp8_h4020pa_fp8_q_pack_kernelEPKDF16bPhii";

// One instance per symbol, constructed on first use.  Construction is what
// registers the code object.  This used to be forced at import time from a
// warmup entry, on the assumption that registering during a stream capture is
// illegal; measured on ROCm 7.2.4 and 7.14 it is not, and every other
// AiterAsmKernel call site in csrc registers lazily too -- asm_mla.cu builds its
// decode kernels this way, inside the path vLLM captures with cudagraphs.
AiterAsmKernel& k_prefill()  { static AiterAsmKernel k(SYM_PREFILL,  CO_PATH); return k; }
AiterAsmKernel& k_q_pack()   { static AiterAsmKernel k(SYM_Q_PACK,   CO_PATH); return k; }

void pa_fp8_q_pack_impl(aiter_tensor_t& q_nope_bf16, aiter_tensor_t& out,
                        hipStream_t stream)
{
    AITER_CHECK(q_nope_bf16.dtype() == AITER_DTYPE_bf16, "q_nope_bf16 must be bf16");
    AITER_CHECK(out.dtype() == AITER_DTYPE_fp8, "out must be fp8");
    AITER_CHECK(q_nope_bf16.dim() == out.dim(), "q and out must have the same rank");
    const int last = q_nope_bf16.dim() - 1;
    AITER_CHECK(static_cast<int>(q_nope_bf16.size(last)) == 448,
                "q_nope_bf16 last dim must be 448 (DSv4 NoPE width)");
    AITER_CHECK(static_cast<int>(out.size(last)) == 512, "out last dim must be 512");
    AITER_CHECK(q_nope_bf16.stride(last) == 1 && out.stride(last) == 1,
                "q and out must be contiguous along the head dim");
    AITER_CHECK(out.is_contiguous(), "out must be contiguous");

    int64_t rows = 1;
    for(int i = 0; i < last; ++i)
        rows *= q_nope_bf16.size(i);
    if(rows == 0)
        return;
    const int src_stride = static_cast<int>(q_nope_bf16.stride(last - 1));

    HipDeviceGuard guard(q_nope_bf16.device_id);
    // 14 quant blocks + 50 pad-zeroing threads = 64
    QPackArgs a{};
    a.src        = q_nope_bf16.data_ptr();
    a.dst        = reinterpret_cast<unsigned char*>(out.data_ptr());
    a.rows       = static_cast<int>(rows);
    a.src_stride = src_stride;
    size_t sz = sizeof(a);
    // 14 quant blocks + 50 pad-zeroing threads = 64
    k_q_pack().launch_kernel({&a, &sz, static_cast<int>(rows), 1, 1, 64, 1, 1, stream});
}

void dsv4_mla_prefill_impl(aiter_tensor_t& q_nope,
                                        aiter_tensor_t& q_rope,
                                        aiter_tensor_t& unified_kv_nope,
                                        aiter_tensor_t& unified_kv_rope,
                                        aiter_tensor_t& kv_indices_prefix,
                                        aiter_tensor_t& kv_indptr_prefix,
                                        aiter_tensor_t& kv_nope,
                                        aiter_tensor_t& kv_rope,
                                        aiter_tensor_t& kv_indices_extend,
                                        aiter_tensor_t& kv_indptr_extend,
                                        aiter_tensor_t& attn_sink,
                                        aiter_tensor_t& kv_max_e,
                                        aiter_tensor_t& out,
                                        float softmax_scale,
                                        aiter_tensor_t& kv_lens_prefix,
                                        aiter_tensor_t& kv_lens_extend,
                                        int kv_stride_q_prefix,
                                        int kv_stride_q_extend,
                                        int page_shift_prefix,
                                        int rows_per_page_prefix,
                                        int scale_off_prefix,
                                        int page_shift_extend,
                                        int rows_per_page_extend,
                                        int scale_off_extend,
                                         hipStream_t stream)
{
    constexpr int D_NOPE_PADDED = PA_FP8_H40_D_NOPE_PADDED;
    constexpr int D_ROPE        = PA_FP8_H40_D_ROPE;
    constexpr int D_HEAD        = PA_FP8_H40_D_HEAD;

    // ---- Shape / dtype validation -----------------------------------------
    AITER_CHECK(q_nope.dim() == 3, "q_nope must be 3-D [N, H, 448], got ndim=", q_nope.dim());
    AITER_CHECK(q_rope.dim() == 3, "q_rope must be 3-D [N, H, 64], got ndim=", q_rope.dim());
    AITER_CHECK(unified_kv_nope.dim() == 2,
                "unified_kv_nope must be 2-D [total_pages, 512], got ndim=", unified_kv_nope.dim());
    AITER_CHECK(unified_kv_rope.dim() == 2,
                "unified_kv_rope must be 2-D [total_pages, 64], got ndim=", unified_kv_rope.dim());
    AITER_CHECK(kv_nope.dim() == 2,
                "kv_nope must be 2-D [total_tokens, 512], got ndim=", kv_nope.dim());
    AITER_CHECK(kv_rope.dim() == 2,
                "kv_rope must be 2-D [total_tokens, 64], got ndim=", kv_rope.dim());
    AITER_CHECK(out.dim() == 3, "out must be 3-D [N, H, 512], got ndim=", out.dim());
    AITER_CHECK(attn_sink.dim() == 1, "attn_sink must be 1-D [H]");

    // Q is packed inside the kernel, so it arrives unpacked as bf16.
    AITER_CHECK(q_nope.dtype() == AITER_DTYPE_bf16, "q_nope must be bf16 [N, H, 448]");
    AITER_CHECK(unified_kv_nope.dtype() == AITER_DTYPE_fp8 && kv_nope.dtype() == AITER_DTYPE_fp8,
                "unified_kv_nope/kv_nope must be fp8");
    AITER_CHECK(q_rope.dtype() == AITER_DTYPE_bf16 && unified_kv_rope.dtype() == AITER_DTYPE_bf16 &&
                    kv_rope.dtype() == AITER_DTYPE_bf16,
                "q_rope/unified_kv_rope/kv_rope must be bf16");
    AITER_CHECK(out.dtype() == AITER_DTYPE_bf16, "out must be bf16");
    AITER_CHECK(attn_sink.dtype() == AITER_DTYPE_fp32, "attn_sink must be fp32");
    AITER_CHECK(kv_max_e.dtype() == AITER_DTYPE_i32, "kv_max_e must be int32");
    AITER_CHECK(kv_max_e.numel() >= 1, "kv_max_e must hold at least one element");

    AITER_CHECK(kv_indptr_prefix.dtype() == AITER_DTYPE_i32, "kv_indptr_prefix must be int32");
    AITER_CHECK(kv_indices_prefix.dtype() == AITER_DTYPE_i32, "kv_indices_prefix must be int32");
    AITER_CHECK(kv_indptr_extend.dtype() == AITER_DTYPE_i32, "kv_indptr_extend must be int32");
    AITER_CHECK(kv_indices_extend.dtype() == AITER_DTYPE_i32, "kv_indices_extend must be int32");

    const int N = static_cast<int>(q_nope.size(0));
    const int H = static_cast<int>(q_nope.size(1));

    // No narrow-head path on purpose -- see the note above this function.
    AITER_CHECK(H >= PA_FP8_MIN_H,
                "dsv4_mla_prefill is only profitable for H >= ",
                PA_FP8_MIN_H, " (the block is ", HEADS_PER_BLOCK,
                " heads wide); got H=", H);

    AITER_CHECK(q_nope.size(2) == 448, "q_nope last dim must be 448 (unpacked bf16 NoPE)");
    AITER_CHECK(q_rope.size(0) == N && q_rope.size(1) == H && q_rope.size(2) == D_ROPE,
                "q_rope shape must be [N, H, 64]");
    AITER_CHECK(unified_kv_nope.size(1) == D_NOPE_PADDED, "unified_kv_nope last dim must be 512");
    AITER_CHECK(unified_kv_rope.size(1) == D_ROPE, "unified_kv_rope last dim must be 64");
    AITER_CHECK(kv_nope.size(1) == D_NOPE_PADDED, "kv_nope last dim must be 512");
    AITER_CHECK(kv_rope.size(1) == D_ROPE, "kv_rope last dim must be 64");
    AITER_CHECK(unified_kv_nope.size(0) == unified_kv_rope.size(0),
                "unified_kv_nope and unified_kv_rope must share total_pages");
    AITER_CHECK(kv_nope.size(0) == kv_rope.size(0),
                "kv_nope and kv_rope must share total_tokens");
    AITER_CHECK(out.size(0) == N && out.size(1) == H && out.size(2) == D_HEAD,
                "out shape must be [N, H, 512]");
    AITER_CHECK(attn_sink.size(0) == H, "attn_sink length must equal H");
    AITER_CHECK(kv_lens_prefix.numel() > 0 || kv_indptr_prefix.size(0) == N + 1,
                "kv_indptr_prefix length must be N+1 unless kv_lens_prefix is given");
    AITER_CHECK(kv_lens_extend.numel() > 0 || kv_indptr_extend.size(0) == N + 1,
                "kv_indptr_extend length must be N+1 unless kv_lens_extend is given");

    // Consecutive query heads inside a tile are indexed by D_NOPE_PADDED /
    // D_ROPE, and the RoPE token stride is derived as H * stride_qr_h, so the
    // Q streams have to be densely packed over both N and H.
    AITER_CHECK(q_nope.stride(2) == 1,
                "q_nope must be unit-stride along the head dim");
    AITER_CHECK(q_nope.stride(0) == static_cast<int64_t>(H) * q_nope.stride(1),
                "q_nope must be densely packed over [N, H]");
    AITER_CHECK(q_rope.stride(2) == 1,
                "q_rope must be unit-stride along the head dim");
    AITER_CHECK(q_rope.stride(0) == static_cast<int64_t>(H) * q_rope.stride(1),
                "q_rope must be densely packed over [N, H]");
    AITER_CHECK(unified_kv_nope.stride(1) == 1 && kv_nope.stride(1) == 1,
                "kv_nope/unified_kv_nope must be contiguous along the head-dim");
    AITER_CHECK(unified_kv_rope.stride(1) == 1 && kv_rope.stride(1) == 1,
                "kv_rope/unified_kv_rope must be contiguous along the head-dim");
    AITER_CHECK(out.stride(2) == 1, "out must be contiguous along the head-dim");

    AITER_CHECK(kv_indices_prefix.is_contiguous() && kv_indptr_prefix.is_contiguous() &&
                    kv_indices_extend.is_contiguous() && kv_indptr_extend.is_contiguous() &&
                    attn_sink.is_contiguous() && kv_max_e.is_contiguous(),
                "kv_indices/kv_indptr (prefix+extend), attn_sink and kv_max_e must be contiguous");

    const int stride_kv_nope_page = static_cast<int>(unified_kv_nope.stride(0));
    const int stride_kv_rope_page = static_cast<int>(unified_kv_rope.stride(0));
    AITER_CHECK(stride_kv_nope_page == static_cast<int>(kv_nope.stride(0)),
                "unified_kv_nope and kv_nope must share row stride");
    AITER_CHECK(stride_kv_rope_page == static_cast<int>(kv_rope.stride(0)),
                "unified_kv_rope and kv_rope must share row stride");

    if(N == 0)
        return;

    // ---- Build kernel args -----------------------------------------------
    PaFp8Kargs kargs{};
    kargs.q_ptr               = q_nope.data_ptr();
    kargs.q_rope_ptr          = q_rope.data_ptr();
    kargs.unified_kv_ptr      = unified_kv_nope.data_ptr();
    kargs.unified_kv_rope_ptr = unified_kv_rope.data_ptr();
    kargs.kv_ptr              = kv_nope.data_ptr();
    kargs.kv_rope_ptr         = kv_rope.data_ptr();
    kargs.attn_sink_ptr       = reinterpret_cast<const float*>(attn_sink.data_ptr());
    kargs.out_ptr             = out.data_ptr();
    kargs.kv_indptr_prefix    = reinterpret_cast<const int*>(kv_indptr_prefix.data_ptr());
    kargs.kv_indices_prefix   = reinterpret_cast<const int*>(kv_indices_prefix.data_ptr());
    kargs.kv_indptr_extend    = reinterpret_cast<const int*>(kv_indptr_extend.data_ptr());
    kargs.kv_indices_extend   = reinterpret_cast<const int*>(kv_indices_extend.data_ptr());
    kargs.N                   = N;
    kargs.H                   = H;
    kargs.D                   = D_HEAD;
    kargs.total_pages         = static_cast<int>(unified_kv_nope.size(0));
    kargs.total_tokens        = static_cast<int>(kv_nope.size(0));
    kargs.stride_q_n          = static_cast<int>(q_nope.stride(0));
    kargs.stride_q_h          = static_cast<int>(q_nope.stride(1));
    kargs.stride_o_n          = static_cast<int>(out.stride(0));
    kargs.stride_o_h          = static_cast<int>(out.stride(1));
    kargs.stride_kv_row       = stride_kv_nope_page;
    kargs.stride_qr_h         = static_cast<int>(q_rope.stride(1));
    kargs.stride_kvr_row      = stride_kv_rope_page;
    // Layout descriptor.  (0, 1, 448) is aiter's own flat [rows, 512] packing;
    // the DSv4 serving pool is (8, bytes_per_page_padded/576, page_size*576).
    const int shifts[2] = {page_shift_prefix, page_shift_extend};
    const int rpp[2]     = {rows_per_page_prefix, rows_per_page_extend};
    const int soff[2]    = {scale_off_prefix, scale_off_extend};
    for(int sg = 0; sg < 2; ++sg)
    {
        AITER_CHECK(shifts[sg] >= 0 && shifts[sg] < 31, "page_shift out of range");
        AITER_CHECK(rpp[sg] >= 1, "rows_per_page must be >= 1");
        AITER_CHECK(soff[sg] >= 0, "scale_off must be >= 0");
        kargs.sgl_page_shift[sg]    = shifts[sg];
        kargs.sgl_rows_per_page[sg] = rpp[sg];
        kargs.sgl_scale_off[sg]     = soff[sg];
    }
    // Dense-index mode is per segment and opt-in: an empty lens tensor keeps
    // that segment on the CSR indptr it has always used.
    const bool dense_p = kv_lens_prefix.numel() > 0;
    const bool dense_e = kv_lens_extend.numel() > 0;
    if(dense_p)
    {
        AITER_CHECK(kv_lens_prefix.dtype() == AITER_DTYPE_i32 &&
                        kv_lens_prefix.is_contiguous(),
                    "kv_lens_prefix must be contiguous int32");
        AITER_CHECK(static_cast<int>(kv_lens_prefix.numel()) >= N,
                    "kv_lens_prefix must hold at least N entries");
        AITER_CHECK(kv_stride_q_prefix > 0, "kv_stride_q_prefix must be > 0 in dense mode");
    }
    if(dense_e)
    {
        AITER_CHECK(kv_lens_extend.dtype() == AITER_DTYPE_i32 &&
                        kv_lens_extend.is_contiguous(),
                    "kv_lens_extend must be contiguous int32");
        AITER_CHECK(static_cast<int>(kv_lens_extend.numel()) >= N,
                    "kv_lens_extend must hold at least N entries");
        AITER_CHECK(kv_stride_q_extend > 0, "kv_stride_q_extend must be > 0 in dense mode");
    }
    kargs.kv_lens_prefix    = dense_p ? reinterpret_cast<const int*>(kv_lens_prefix.data_ptr())
                                      : nullptr;
    kargs.kv_lens_extend    = dense_e ? reinterpret_cast<const int*>(kv_lens_extend.data_ptr())
                                      : nullptr;
    kargs.kv_stride_q_prefix = kv_stride_q_prefix;
    kargs.kv_stride_q_extend = kv_stride_q_extend;
    kargs.softmax_scale       = softmax_scale;
    // This build is bound to the flat KV layout (PA_SGLANG_PAGED 0), so a paged
    // page grid would be read as if it were flat -- wrong results, not an error.
    // Reject it instead; the layout arguments stay in the signature so a paged
    // build is a recompile rather than an API change.
    kargs.max_e_ptr           = reinterpret_cast<const int*>(kv_max_e.data_ptr());

    // ---- Launch ----------------------------------------------------------
    HipDeviceGuard guard(q_nope.device_id);

    const int num_h_blocks = ceil_div(H, HEADS_PER_BLOCK);
    dim3 grid(N, num_h_blocks, 1);
    dim3 block(BLOCK_SIZE);
    size_t sz = sizeof(kargs);
    k_prefill().launch_kernel({&kargs, &sz,
                               N, num_h_blocks, 1,
                               BLOCK_SIZE, 1, 1,
                               stream});
}

} // namespace

AITER_CTYPES_ERROR_DEF

// The impls above validate with AITER_CHECK, which throws, and an exception
// must not cross the extern "C" boundary.  aiter_safe_call parks the message in
// the module's TLS slot and returns -1; jit/core.py's ctypes binding picks it up
// and raises RuntimeError with the original text, so the diagnostics survive.

#define PA_H40_CO_ENTRY(call, name)                                               \
    return aiter_safe_call(g_aiter_last_error, [&] {                              \
        call;                                                                     \
        return 0;                                                                 \
    });

AITER_C_ITFS int dsv4_mla_q_pack_fwd(aiter_tensor_t* q_nope_bf16,
                                  aiter_tensor_t* out,
                                  void* stream)
{
    PA_H40_CO_ENTRY(pa_fp8_q_pack_impl(*q_nope_bf16, *out,
                                       static_cast<hipStream_t>(stream)),
                    "pa_fp8_q_pack")
}

AITER_C_ITFS int dsv4_mla_prefill_fwd(aiter_tensor_t* q_nope,
                                                   aiter_tensor_t* q_rope,
                                                   aiter_tensor_t* unified_kv_nope,
                                                   aiter_tensor_t* unified_kv_rope,
                                                   aiter_tensor_t* kv_indices_prefix,
                                                   aiter_tensor_t* kv_indptr_prefix,
                                                   aiter_tensor_t* kv_nope,
                                                   aiter_tensor_t* kv_rope,
                                                   aiter_tensor_t* kv_indices_extend,
                                                   aiter_tensor_t* kv_indptr_extend,
                                                   aiter_tensor_t* attn_sink,
                                                   aiter_tensor_t* kv_max_e,
                                                   aiter_tensor_t* out,
                                                   float softmax_scale,
                                                   aiter_tensor_t* kv_lens_prefix,
                                                   aiter_tensor_t* kv_lens_extend,
                                                   int64_t kv_stride_q_prefix,
                                                   int64_t kv_stride_q_extend,
                                                   int64_t page_shift_prefix,
                                                   int64_t rows_per_page_prefix,
                                                   int64_t scale_off_prefix,
                                                   int64_t page_shift_extend,
                                                   int64_t rows_per_page_extend,
                                                   int64_t scale_off_extend,
                                                   void* stream)
{
    PA_H40_CO_ENTRY(
        dsv4_mla_prefill_impl(
            *q_nope, *q_rope, *unified_kv_nope, *unified_kv_rope,
            *kv_indices_prefix, *kv_indptr_prefix, *kv_nope, *kv_rope,
            *kv_indices_extend, *kv_indptr_extend, *attn_sink, *kv_max_e, *out,
            softmax_scale, *kv_lens_prefix, *kv_lens_extend,
            static_cast<int>(kv_stride_q_prefix), static_cast<int>(kv_stride_q_extend),
            static_cast<int>(page_shift_prefix),
            static_cast<int>(rows_per_page_prefix),
            static_cast<int>(scale_off_prefix),
            static_cast<int>(page_shift_extend),
            static_cast<int>(rows_per_page_extend),
            static_cast<int>(scale_off_extend),
            static_cast<hipStream_t>(stream)),
        "dsv4_mla_prefill")
}
