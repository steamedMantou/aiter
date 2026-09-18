// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// FP8 sparse paged attention for DeepSeek-V4 on gfx950.
//
// Four waves cover 128 heads.  Q and staged KV use scaled FP8 MFMA, while
// RoPE remains BF16.  Persistent PV accumulators stay in AGPRs; LDS provides
// the accumulator initialization, AGPR-to-VGPR bridge, and output transpose.
//
// Input layout:
//   q_nope           : [N, H, 448] BF16, packed to FP8 in the prologue
//   kv_nope          : FP8 NoPE values plus E8M0 block scales
//   q_rope / kv_rope : [.., 64] BF16
//
// KV block scales are normalized to one exponent per token in LDS.  QK routes
// Q/K exponents through MFMA scale operands.  PV folds the token exponent into
// the FP8 probability operand and accumulates in FP32.
#pragma once

#include <opus/opus.hpp>
#include <bit>
#include <type_traits>

// The whole kernel lives in its own namespace: aiter's pa_sparse_prefill_opus.h
// already defines a `pa_fp8_kargs` in the same translation unit, and its IMPL
// section puts bf16_t / fp8_t at global scope.
namespace pa_sparse_mla {

// Minimum admitted head count.  Keep in sync with the Python dispatcher.
#ifndef PA_SPARSE_MLA_MIN_H
#define PA_SPARSE_MLA_MIN_H 16
#endif

// ---------------------------------------------------------------------------
// Kernel arguments
// ---------------------------------------------------------------------------
struct pa_fp8_kargs
{
    const void* __restrict__ q_ptr;             // [N, H, 448]          bf16, packed in-kernel
    const void* __restrict__ q_rope_ptr;        // [N, H, 64]           bf16
    const void* __restrict__ unified_kv_ptr;    // [total_pages, 512]   packed fp8
    const void* __restrict__ unified_kv_rope_ptr;  // [total_pages, 64] bf16
    const void* __restrict__ kv_ptr;            // [total_tokens, 512]  packed fp8
    const void* __restrict__ kv_rope_ptr;       // [total_tokens, 64]   bf16
    const float* __restrict__ attn_sink_ptr;        // [H]               fp32
    void* __restrict__ out_ptr;                     // [N, H, D]         bf16
    const int* __restrict__ kv_indptr_prefix;
    const int* __restrict__ kv_indices_prefix;
    const int* __restrict__ kv_indptr_extend;
    const int* __restrict__ kv_indices_extend;
    int N;
    int H;
    int D;
    int total_pages;
    int total_tokens;
    int stride_q_n;    // elements (fp8)
    int stride_q_h;
    int stride_o_n;    // elements (bf16)
    int stride_o_h;
    int stride_kv_row;  // elements (fp8)
    int stride_qr_h;    // elements (bf16)
    int stride_kvr_row; // elements (bf16)
    // sglang DSv4 pool descriptor.  page_shift == 0 selects the legacy flat
    // [rows, 512] layout; otherwise the buffer is a uniform stride_kv_row grid
    // in which page p's token j is grid row p*rows_per_page + j, and the
    // per-token E8M0 lives at p*rows_per_page*stride_kv_row + scale_off + j*8.
    // Prefix and extend may use different page descriptors.
    int   sgl_page_shift[2];      // [0] prefix / unified, [1] extend
    int   sgl_rows_per_page[2];
    int   sgl_scale_off[2];
    // Dense-index mode: a non-null lens selects q*stride addressing and ignores
    // the corresponding CSR indptr.
    const int* __restrict__ kv_lens_prefix;
    const int* __restrict__ kv_lens_extend;
    int   kv_stride_q_prefix;
    int   kv_stride_q_extend;
    float softmax_scale;
    // Max per-token E8M0 exponent over the whole gathered KV, as a *device*
    // pointer -- a host scalar would force a D2H sync per call and break graph
    // capture.  Unread under PA_NO_COLLAPSE, which takes the frame from each
    // tile instead; the field stays so the call signature does not move.
    const int* __restrict__ max_e_ptr;
    // Optional workgroup-to-query permutation used for XCD placement.  It must
    // be a permutation of [0, N); null selects reverse query order.
    const int* __restrict__ row_map;
};

struct pa_fp8_traits
{
#ifndef PA_SCHED_REP
#define PA_SCHED_REP 3
#endif
#ifndef PA_SCHED_DS
#define PA_SCHED_DS 3
#endif
#ifndef PA_SCHED_MFMA
#define PA_SCHED_MFMA 4
#endif
    static constexpr int Q_TILE     = 16;   // heads per mfma tile
#ifndef PA_Q_SUB
#define PA_Q_SUB 2
#endif
#ifndef PA_NUM_WARPS
#define PA_NUM_WARPS 4
#endif
    static constexpr int Q_SUB      = PA_Q_SUB;  // mfma tiles per wave
    static constexpr int KV_TILE    = 128;  // tokens per LDS tile (== mfma K)
    static constexpr int D_TILE     = 512;
    static constexpr int NUM_WARPS  = PA_NUM_WARPS;
    static constexpr int WARP_SIZE  = 64;
    static constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE;              // 256
    static constexpr int HEADS_PER_BLOCK = NUM_WARPS * Q_SUB * Q_TILE;    // 128

    static constexpr int D_NOPE = 448;              // real NoPE width
    static constexpr int NBLK   = D_NOPE / 32;      // 14 E8M0 blocks per row
    static constexpr int N_TILES  = KV_TILE / 16;   // 8  QK n-subtiles
    static constexpr int D_SLICES = D_TILE / 128;   // 4  QK k-slices (448 padded to 512)
    static constexpr int O_TILES  = D_NOPE / 16;    // 28 PV output subtiles
    static constexpr int K_BLOCKS = KV_TILE / 32;   // 4  tr_b8 reads per PV operand
    // Transpose the MFMA C layout through LDS before coalesced global stores.
#ifndef PA_EPI_LDS
#define PA_EPI_LDS 1
#endif
    // BF16 values per staged row; padding spreads rows across LDS banks.
#ifndef PA_EPI_PITCH
#define PA_EPI_PITCH 520
#endif
#ifndef PA_NO_COLLAPSE
// Keep the cache unchanged and derive the FP8 probability frame per LDS tile.
#define PA_NO_COLLAPSE 1
#endif
#if PA_NO_COLLAPSE
#undef PA_LDS_REQUANT
#define PA_LDS_REQUANT 1
#endif
#ifndef PA_LDS_REQUANT
// Normalize each staged token to one exponent before QK/PV consume the tile.
#define PA_LDS_REQUANT 0
#endif
// Streaming cache policy: keep Q reusable and mark output stores non-temporal.
// PV partial waits require an LGKM queue containing only this batch's
// transposed LDS loads; the build-time ISA guard enforces that condition.
#ifndef PA_PV_PARTIAL_WAIT
#define PA_PV_PARTIAL_WAIT 1
#endif
#ifndef PA_NT_Q
#define PA_NT_Q 0
#endif
#ifndef PA_NT_OUT
#define PA_NT_OUT 1
#endif
#define PA_NT_AUX  (PA_NT_Q   ? 2 : 0)
#define PA_NT_AUXO (PA_NT_OUT ? 2 : 0)
#ifndef PA_RQ_SPLIT
#define PA_RQ_SPLIT 2
#endif
// Optional per-lane k==0 skip.  Disabled because it is divergent and changes
// subnormal handling: the conversion at k==0 is not an identity.
#ifndef PA_RQ_BRANCH
#define PA_RQ_BRANCH 0
#endif
// Number of ATOM LDS reads issued before requant conversion begins.  It must
// divide O_TILES/RQ_SPLIT; legal values are 1, 2, 7, and 14.
// Fold each tile's exponent maximum in registers before one LDS atomic per wave.
#ifndef PA_RQ_FOLD
#define PA_RQ_FOLD 1
#endif
#ifndef PA_MFMA_ASM_VOL
#define PA_MFMA_ASM_VOL volatile
#endif
#ifndef PA_MFMA_ASM_VOL
#define PA_MFMA_ASM_VOL volatile
#endif
#ifndef PA_MFMA_ASM
#define PA_MFMA_ASM 0
#endif
#ifndef PA_MFMA_ASM_FILL
#define PA_MFMA_ASM_FILL "s_nop 0\n\t"
#endif
#ifndef PA_MFMA_NOP
#define PA_MFMA_NOP 0
#endif
#ifndef PA_SCHED_MODE
#define PA_SCHED_MODE 0
#endif
#ifndef PA_GATHER_MODE
#define PA_GATHER_MODE 0
#endif
// Requantize with the hardware FP8->BF16->FP8 scaled conversion path.
#ifndef PA_RQ_CVT
#define PA_RQ_CVT 1
#endif
// Diagnostic: keep LDS traffic but replace requantization with identity.
// Numerically invalid.
#ifndef PA_RQ_NOOP
#define PA_RQ_NOOP 0
#endif
// Diagnostic: skip the requant LDS read-modify-write.  Numerically invalid.
#ifndef PA_RQ_SKIP
#define PA_RQ_SKIP 0
#endif
// Compute paged row indices in 32 bits and widen once, instead of carrying the
// whole address chain in 64.  Bit-identical; 0 restores the old expressions.
#ifndef PA_ADDR32
#define PA_ADDR32 3
#endif
// Build P's per-token power-of-two scale straight out of the E8M0 byte instead
// of going through an integer exponent and v_ldexp_f32, and let the fp8 pack's
// own scale operand carry the tile-uniform remainder.  Every factor is an exact
// power of two, so the result is bit-identical.
//   1 = multiply form   2 = same, but the pack's scale is treated as a divisor
#ifndef PA_P_SCALE_MUL
#define PA_P_SCALE_MUL 1
#endif
// Leave the fp8 pack's `old` operand uninitialised.  Both halves are written, so
// the zero it currently gets is a dead v_mov_b32 per packed dword.
#ifndef PA_PACK_UNDEF
#define PA_PACK_UNDEF 1
#endif
// Optionally overlap tile n+1 requantization with tile n PV.
#ifndef PA_RQ_PIPELINE
#define PA_RQ_PIPELINE 0
#endif
// Stage each E8M0 exponent as its FP32 power-of-two bit pattern.
#ifndef PA_ETOK_F32
#define PA_ETOK_F32 1
#endif
#if PA_ETOK_F32 && !PA_P_SCALE_MUL
#error "PA_ETOK_F32 needs PA_P_SCALE_MUL"
#endif
#if PA_ETOK_F32 && PA_RQ_PIPELINE
// The pipelined cold path does not initialize the FP32 exponent table.
#error "PA_ETOK_F32 and PA_RQ_PIPELINE are not wired together"
#endif
// Diagnostic one-shot prologue instructions.
#ifndef PA_ICACHE_PROBE
#define PA_ICACHE_PROBE 0
#endif
// Which tile_max_e slot a buffer uses.  Pipelined, tile n reads its own frame
// while its PV is already writing tile n+1's, so the two must not alias.
#define PA_ME_SLOT(b) (PA_RQ_PIPELINE ? (b) : 0)
#ifndef PA_ROPE_LATE
#define PA_ROPE_LATE 0
#endif
#ifndef PA_GATHER_SPLIT
#define PA_GATHER_SPLIT 2   // slots gathered above the requant pass; 4 = all (original)
#endif
#ifndef PA_RQ_SERIAL
#define PA_RQ_SERIAL 0
#endif
#ifndef PA_RQ_AGPR
#define PA_RQ_AGPR 0
#endif
// Keep persistent PV accumulators in AGPRs and bridge through LDS.
#ifndef PA_ACC_LDS_BRIDGE
#define PA_ACC_LDS_BRIDGE 1
#endif
#ifndef PA_ACC_LDS_INIT
#define PA_ACC_LDS_INIT PA_ACC_LDS_BRIDGE
#endif
#ifndef PA_ACC_LDS_EPI
#define PA_ACC_LDS_EPI PA_ACC_LDS_BRIDGE
#endif
#ifndef PA_ACC_EPI_PITCH
#define PA_ACC_EPI_PITCH 516
#endif
#ifndef PA_ACC_INIT_BATCH
#define PA_ACC_INIT_BATCH 8
#endif
#ifndef PA_ACC_INIT_KEEP
#define PA_ACC_INIT_KEEP 6
#endif
#ifndef PA_ACC_INIT_REFILL
#define PA_ACC_INIT_REFILL 2
#endif
#ifndef PA_ACC_EPI_PIPE
#define PA_ACC_EPI_PIPE 1
#endif
#ifndef PA_RQ_BATCH
#define PA_RQ_BATCH 14
#endif
#ifndef PA_PV_BATCH
#define PA_PV_BATCH 4
#endif
    // o_tiles whose V is fetched before the batch's mfmas run.  Must divide
    // O_TILES (28): 1, 2, 4, 7, 14, 28.
    static constexpr int PV_BATCH = PA_PV_BATCH;

    static constexpr int S_LEN = N_TILES * 4;       // 32 scores per lane per tile
    static constexpr int O_LEN = D_TILE / 16;       // 32 f32x4 accumulators per lane per tile

    // LDS tile geometry.  A 16-byte "cell" is (one token, 16 consecutive d).
    // Eight cells for eight consecutive token slots form a 128-byte "atom",
    // which is exactly what one ds_read_b64_tr_b8 consumes.
    static constexpr int ATOM = 128;                 // bytes
// Split the next-tile copy around requantization to distribute LDS writes.
// The prologue exponent gather uses independent address/result registers and
// an explicit VMEM drain; in-tile gathers remain compiler scheduled.
#ifndef PA_EXP_GATHER_ASM
#define PA_EXP_GATHER_ASM 1
#endif
#ifndef PA_COPY_SPLIT
#define PA_COPY_SPLIT 2
#endif
#ifndef PA_TOP_VMCNT
// Leave the four youngest RoPE copies in flight across NoPE requantization.
// They are drained before the first QK RoPE read.
#define PA_TOP_VMCNT 4
#endif
#ifndef PA_PAD
#define PA_PAD 64
#endif
    // Padding between slot blocks.
    static constexpr int PAD  = PA_PAD;
    static constexpr int ROW  = 32 * ATOM + PAD;     // 4160 bytes per slot block
    static constexpr int TILE_BYTES = 16 * ROW;      // 66560
    static constexpr int SLOTS_PER_WAVE = 16 / NUM_WARPS;   // 4 slot blocks staged per wave

// Must sit above its first use.  Read below one, the `#if` silently takes the
// false arm, so EXP_BYTES gets the flat value while every later branch gets the
// paged one -- a mixed build whose exponent buffer is sized 0 and written past.
#ifndef PA_SGLANG_PAGED
#define PA_SGLANG_PAGED 1
#endif

    // Per tile: the 16 packed scale bytes of each token, plus the derived
    // per-token exponent in two views (plain for P, transposed for QK).
#if PA_SGLANG_PAGED
    // sglang's row has RoPE bf16 where aiter's has the E8M0 bytes, so the
    // exponent cannot ride the tile.  It comes from the page's own scale region
    // instead: one byte per token on the same two-tile-ahead pipeline the old
    // per-32 gather used, at 1/16th of its LDS (128 B/buffer against 2048).
    static constexpr int EXP_BYTES  = KV_TILE * 8;    // 7 UE8M0 + pad per token
#else
    static constexpr int EXP_BYTES  = 0;              // exponent read from the tile
#endif
    // s_etok / s_etok_t live *inside* the KV tile, in cells 29 and 30 of slot
    // block 0.  Cells 28..31 hold the packed E8M0 bytes and the pad; since the
    // QK stopped reading its tail slice nothing reads 29..31 any more, and cell
    // k of slot block 0 is 128 contiguous bytes (slot s_lo at s_lo*16) -- exactly
    // one of these tables.  Costs no LDS and comes double-buffered for free.
    static constexpr int ETOK_OFF = 29 * ATOM;        // s_etok[p]
    static constexpr int ETKT_OFF = 30 * ATOM;        // s_etok_t[c*8+nt]
    // 512 B of f32 scales, one per token.  Slot block 0 only has cells 28 and
    // 31 spare, so this goes in slot block 1's tail four -- inside TILE_BYTES
    // already, so it costs no LDS.
    static constexpr int ETOKF_OFF = ROW + 28 * ATOM; // s_etok_f[p], f32
    static constexpr int SCALE_BYTES = EXP_BYTES;
#ifndef PA_ROPE_BUFS
// One RoPE buffer fits beside the two NoPE tiles and scale buffers.
#define PA_ROPE_BUFS 1
#endif
    static constexpr int ROPE_BUFS = PA_ROPE_BUFS;
// page_shift=0, rows_per_page=1, scale_off=448 describes the flat layout.
// Other descriptors select the paged 576-byte layout and page-tail scales.
#ifndef PA_QK_PIPE
// Keep K operands prefetched across n-subtiles.
#define PA_QK_PIPE ((EXP_BYTES <= KV_TILE * 8) ? 2 : 1)
#endif
#ifndef PA_QK_PIPE_K
#define PA_QK_PIPE_K PA_QK_PIPE
#endif
#ifndef PA_QK_PIPE_R
#define PA_QK_PIPE_R PA_QK_PIPE
#endif
// Prefetch one NoPE K d-slice; load the remaining slices in place.
#ifndef PA_QK_PREFETCH_K
#define PA_QK_PREFETCH_K 1
#endif
    static constexpr int QK_PIPE_K = PA_QK_PIPE_K;
    static constexpr int QK_PIPE_R = PA_QK_PIPE_R;
    static constexpr bool SGL_PAGED = (PA_SGLANG_PAGED != 0);

    // Staging requantisation: BLOCK_SIZE threads over KV_TILE tokens.
    static constexpr int REQ_SPLIT = BLOCK_SIZE / KV_TILE;  // 2 threads per token
    static constexpr int RQ_SPLIT = PA_LDS_REQUANT ? PA_RQ_SPLIT : 1;
    static constexpr int RQ_ATOMS = O_TILES / RQ_SPLIT;
    static constexpr int RQ_BATCH = PA_RQ_BATCH;
    static_assert(RQ_ATOMS % RQ_BATCH == 0, "RQ_BATCH must divide O_TILES/RQ_SPLIT");
    static_assert(RQ_SPLIT * KV_TILE <= BLOCK_SIZE, "not enough threads");
    static_assert(O_TILES % RQ_SPLIT == 0, "RQ_SPLIT must divide O_TILES");
    static constexpr int REQ_BLKS  = NBLK / REQ_SPLIT;      // 7 E8M0 blocks each
    static constexpr int REQ_TAIL  = (32 - O_TILES) / REQ_SPLIT;  // 2 tail cells each

    // RoPE stays bf16.  ds_read_b64_tr_b16 transposes a 4x16 u16 tile, so the
    // atom is 4 token slots x 16 bf16 and the token mapping is the identity
    // (slot_hi = t/4, slot_lo = t%4) -- no permutation, unlike the fp8 side.
    static constexpr int D_ROPE   = 64;
    static constexpr int ROPE_KST = D_ROPE / 32;          // 2 QK bf16 k-steps
    static constexpr int O_TILES_R = D_ROPE / 16;         // 4 PV output subtiles
    static constexpr int PV_R_KST = KV_TILE / 32;         // 4 PV bf16 k-steps
    static constexpr int ROPE_ROWS = KV_TILE / 4;         // 32 slot blocks
    static constexpr int RROWS_PER_WAVE = ROPE_ROWS / NUM_WARPS;  // 8
    // 640 is the only stride that makes the QK-RoPE read conflict-free: it is
    // 128 mod 256, so consecutive slot blocks alternate between bank halves and
    // the 64 lanes cover the 256-byte period exactly four times.
#ifndef PA_PAD_R
#define PA_PAD_R 0
#endif
    // Unpadded, so two consecutive slot blocks are contiguous and one
    // 16-B/lane async copy fills both.
    static constexpr int ROW_R = 4 * 128 + PA_PAD_R;      // 512
    static_assert(PA_PAD_R == 0, "the rope copy assumes contiguous slot blocks");
    static constexpr int RCOPY = ROPE_ROWS / (2 * NUM_WARPS);  // 4 copies per wave
    static constexpr int ROPE_BYTES = ROPE_ROWS * ROW_R;  // 16384

    static constexpr size_t smem_size_bytes()
    {
        return 2 * (size_t)TILE_BYTES + 2 * (size_t)SCALE_BYTES
             + (size_t)ROPE_BUFS * (size_t)ROPE_BYTES;
    }
};

__host__ __device__ inline int pa_fp8_ceil_div(int a, int b) { return (a + b - 1) / b; }

// Named for what it computes, not for one of the two phases it serves.  The
// same object runs prefill and decode -- the adapters differ only in the
// tensors they hand it -- so a decode dispatch used to appear in a trace as
// "pa_prefill_fp8_kernel", which reads like the wrong kernel was captured.
__global__ void pa_sparse_mla_kernel(pa_fp8_kargs kargs);



#if !defined(__HIP_DEVICE_COMPILE__) || !defined(__gfx950__)
__global__ void pa_sparse_mla_kernel(pa_fp8_kargs) {}

#else

namespace pa_fp8 {

using namespace opus;

using bf16_t = __bf16;
using i32x2  = int __attribute__((ext_vector_type(2)));
using i32x4  = int __attribute__((ext_vector_type(4)));
using i32x8  = int __attribute__((ext_vector_type(8)));
using f32x4  = float __attribute__((ext_vector_type(4)));
using bf16x8 = __bf16 __attribute__((ext_vector_type(8)));

constexpr float LOG2_E = 1.44269504089f;

// ---------------------------------------------------------------------------
// LDS addressing.  Tile-local token position p in [0,128) lives at slot
//     s_hi = 4*(p/32) + ((p%16)/4)   in [0,16)
//     s_lo = 4*((p%32)/16) + (p%4)   in [0,8)
// This permutation is exactly what makes one tr_b8 atom deliver the eight
// token positions that one PV mfma operand block wants.
// ---------------------------------------------------------------------------
__device__ inline float reduce_max_groups(float v)
{
    vector_t<u32_t, 2> r32 = __builtin_amdgcn_permlane32_swap(
        std::bit_cast<u32_t>(v), std::bit_cast<u32_t>(v), false, true);
    v = max(std::bit_cast<float>(r32.x), std::bit_cast<float>(r32.y));
    vector_t<u32_t, 2> r16 = __builtin_amdgcn_permlane16_swap(
        std::bit_cast<u32_t>(v), std::bit_cast<u32_t>(v), false, true);
    return max(std::bit_cast<float>(r16.x), std::bit_cast<float>(r16.y));
}

// Integer twin of reduce_max_groups: permlane32/16_swap fold lane bits 4 and 5,
// which is exactly g = lane >> 4.  In registers throughout -- no LDS, no extra
// live value, which matters on a kernel with no register headroom at all.
__device__ inline int reduce_max_groups_i(int v)
{
    vector_t<u32_t, 2> r32 = __builtin_amdgcn_permlane32_swap(
        (u32_t)v, (u32_t)v, false, true);
    v = max((int)r32.x, (int)r32.y);
    vector_t<u32_t, 2> r16 = __builtin_amdgcn_permlane16_swap(
        (u32_t)v, (u32_t)v, false, true);
    return max((int)r16.x, (int)r16.y);
}

// Fold the remaining 16 lanes of a row with DPP row_shr, so a whole wave ends
// up with one value in lane 15.  `old` is v itself and bound_ctrl is off, so an
// out-of-range source leaves v alone and max() is unaffected.  Pure VALU.
__device__ inline int reduce_max_row16_i(int v)
{
    v = max(v, __builtin_amdgcn_update_dpp(v, v, 0x111, 0xf, 0xf, false));
    v = max(v, __builtin_amdgcn_update_dpp(v, v, 0x112, 0xf, 0xf, false));
    v = max(v, __builtin_amdgcn_update_dpp(v, v, 0x114, 0xf, 0xf, false));
    v = max(v, __builtin_amdgcn_update_dpp(v, v, 0x118, 0xf, 0xf, false));
    return v;
}

__device__ inline float reduce_sum_groups(float v)
{
    vector_t<u32_t, 2> r32 = __builtin_amdgcn_permlane32_swap(
        std::bit_cast<u32_t>(v), std::bit_cast<u32_t>(v), false, true);
    v = std::bit_cast<float>(r32.x) + std::bit_cast<float>(r32.y);
    vector_t<u32_t, 2> r16 = __builtin_amdgcn_permlane16_swap(
        std::bit_cast<u32_t>(v), std::bit_cast<u32_t>(v), false, true);
    return std::bit_cast<float>(r16.x) + std::bit_cast<float>(r16.y);
}

__device__ inline f32x4 mma(const i32x8& a, const i32x8& b, const f32x4& c)
{
    // fmt 0/0 = e4m3 x e4m3; scale exponents 127 = 2^0 (scaling is done in VALU)
    return __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a, b, c, 0, 0, 0, 127, 0, 127);
}

// PV with a runtime scale_b.  The B operand is P and its lane index c is the
// head, so a *per-head* power of two rides here for free; the scale's lane
// sharing runs across g (token groups) at fixed c, i.e. within one head.
// scale_a could not carry it -- that side is V, indexed along d.
__device__ inline f32x4 mma_sb(const i32x8& a, const i32x8& b, const f32x4& c, int sb)
{
    return __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a, b, c, 0, 0, 0, 127, 0, sb);
}

// QK: scale_a carries K's per-token E8M0 (one byte, op_sel 0), scale_b carries
// Q's per-32-block E8M0 (four bytes, op_sel picks the slice).  Both are applied
// by the hardware, so the dequantisation costs no VALU at all.
template <int OPSEL_B>
__device__ inline f32x4 mma_qk(const i32x8& a, const i32x8& b, const f32x4& c,
                               int sa, int sb)
{
    return __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
        a, b, c, 0, 0, 0, sa, OPSEL_B, sb);
}

// One dword = 4 fp8 bytes inside one 32-element block, so k is uniform.
// Rescaling e4m3 by 2^-k is a subtraction of k from the exponent field; bytes
// whose exponent would fall to <= 0 are flushed (their magnitude is < 2^-7 of
// the token max).
// Hardware path: FP8 -> BF16 scaled by 2^-k -> FP8.  It preserves gradual
// underflow, unlike the SWAR helper below which flushes underflow to zero.
typedef __bf16 pa_bf16_t;
typedef pa_bf16_t pa_bf16x2 __attribute__((ext_vector_type(2)));
typedef short     pa_s16x2  __attribute__((ext_vector_type(2)));
__device__ inline unsigned shift_exp_dword_cvt(unsigned dw, int k)
{
    // 2^-k as an f32 bit pattern; k is per-token, shared by the ATOM's 4 dwords
    const float sc = __builtin_bit_cast(float, (127 - k) << 23);
    pa_bf16x2 lo = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp8(dw, sc, false);
    pa_bf16x2 hi = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp8(dw, sc, true);
    // Both pack operations overwrite their selected half; the initial value is
    // intentionally undefined to avoid materializing a dead seed.
#if PA_PACK_UNDEF
    pa_s16x2 r;
#else
    pa_s16x2 r = {0, 0};
#endif
    r = __builtin_amdgcn_cvt_scalef32_pk_fp8_bf16(r, lo, 1.0f, false);
    r = __builtin_amdgcn_cvt_scalef32_pk_fp8_bf16(r, hi, 1.0f, true);
    return __builtin_bit_cast(unsigned, r);
}
// Pack four f32 into one dword of e4m3, dividing by a common power of two.
// v_cvt_scalef32_pk_fp8_f32 applies the scale for free, which is what lets the
// per-element ldexp collapse into a plain multiply by the E8M0 byte reread as
// an f32 exponent field.
__device__ inline int pack_fp8_scaled(float p0, float p1, float p2, float p3,
                                      float s)
{
    // Same as shift_exp_dword_cvt: both halves are written, so the init is dead
    // work.  See the note there about leaving it uninitialised.
#if PA_PACK_UNDEF
    pa_s16x2 r;
#else
    pa_s16x2 r = {0, 0};
#endif
    r = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(r, p0, p1, s, false);
    r = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(r, p2, p3, s, true);
    return __builtin_bit_cast(int, r);
}
__device__ inline unsigned shift_exp_dword(unsigned dw, int k)
{
    const unsigned h = 0x80808080u;
    const unsigned S = (unsigned)(8 * k) * 0x01010101u;
    const unsigned T = (unsigned)(8 * (k + 1)) * 0x01010101u;
    const unsigned u = dw & 0x7f7f7f7fu;
    const unsigned sgn = dw & h;
    const unsigned uh = u | h;
    const unsigned d = (uh - S) & 0x7f7f7f7fu;
    const unsigned mh = (uh - T) & h;
    const unsigned m = mh - (mh >> 7);
    return (d & m) | (sgn & mh);
}

__device__ inline int slot_hi_of(int p) { return 4 * (p >> 5) + ((p & 15) >> 2); }
__device__ inline int slot_lo_of(int p) { return 4 * ((p & 31) >> 4) + (p & 3); }

// RoPE keeps bf16 on both sides, so it runs on the plain 16x16x32 tile.
__device__ inline f32x4 mma_bf16(const bf16x8& a, const bf16x8& b, const f32x4& c)
{
#if PA_ACC_LDS_BRIDGE
    f32x4 d = c;
    asm volatile(
        "v_mfma_f32_16x16x32_bf16 %0, %1, %2, %0"
        : "+v"(d)
        : "v"(a), "v"(b));
    return d;
#else
    return __builtin_amdgcn_mfma_f32_16x16x32_bf16(a, b, c, 0, 0, 0);
#endif
}

#if PA_ACC_LDS_BRIDGE
__device__ inline void pin_acc_vgpr(f32x4&) {}

__device__ inline f32x4 mma_sb_acc(const i32x8& a, const i32x8& b, f32x4 c, int sb)
{
    const int sa = 127;
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 "
        "%0, %1, %2, %0, %3, %4 op_sel_hi:[0,0,0]"
        : "+a"(c)
        : "v"(a), "v"(b), "v"(sa), "v"(sb));
    return c;
}

__device__ inline f32x4 mma_bf16_acc(const bf16x8& a, const bf16x8& b, f32x4 c)
{
    asm volatile(
        "v_mfma_f32_16x16x32_bf16 %0, %1, %2, %0"
        : "+a"(c)
        : "v"(a), "v"(b));
    return c;
}

template <int OFFSET = 0>
__device__ inline void ds_write_acc(u32_t addr, const f32x4& v)
{
    static_assert(OFFSET >= 0 && OFFSET <= 0xffff, "DS offset out of range");
    asm volatile("ds_write_b128 %0, %1 offset:%2"
                 :: "v"(addr), "a"(v), "n"(OFFSET) : "memory");
}

__device__ inline void ds_write_vgpr(u32_t addr, const f32x4& v)
{
    asm volatile("ds_write_b128 %0, %1" :: "v"(addr), "v"(v) : "memory");
}

__device__ inline f32x4 ds_read_acc(u32_t addr)
{
    f32x4 v;
    asm volatile("ds_read_b128 %0, %1" : "=a"(v) : "v"(addr) : "memory");
    return v;
}

template <int OFFSET = 0>
__device__ inline f32x4 ds_read_vgpr(u32_t addr)
{
    static_assert(OFFSET >= 0 && OFFSET <= 0xffff, "DS offset out of range");
    f32x4 v;
    asm volatile("ds_read_b128 %0, %1 offset:%2"
                 : "=v"(v) : "v"(addr), "n"(OFFSET) : "memory");
    return v;
}
#else
__device__ inline void pin_acc_vgpr(f32x4&) {}

__device__ inline f32x4 mma_sb_acc(const i32x8& a, const i32x8& b,
                                   const f32x4& c, int sb)
{
    return mma_sb(a, b, c, sb);
}

__device__ inline f32x4 mma_bf16_acc(const bf16x8& a, const bf16x8& b,
                                     const f32x4& c)
{
    return mma_bf16(a, b, c);
}
#endif

// Hand-scheduled RoPE MFMA pair.  Separate C/D operands break the WAR edge;
// srcB remains in AGPRs and PA_MFMA_ASM_FILL separates the two issues.
__device__ inline void mma_bf16_pair(f32x4& d0, f32x4& d1,
                                     const f32x4& c0, const f32x4& c1,
                                     const bf16x8& a, const bf16x8& b0,
                                     const bf16x8& b1)
{
    asm volatile("v_mfma_f32_16x16x32_bf16 %0, %4, %5, %2\n\t"
                 PA_MFMA_ASM_FILL
                 "v_mfma_f32_16x16x32_bf16 %1, %4, %6, %3"
                 : "=v"(d0), "=v"(d1)
                 : "v"(c0), "v"(c1), "v"(a), "a"(b0), "a"(b1));
}

__device__ inline i32x8 pack32(const i32x4& lo, const i32x4& hi)
{
    i32x8 r;
#pragma unroll
    for (int t = 0; t < 4; ++t) { r[t] = lo[t]; r[4 + t] = hi[t]; }
    return r;
}

__device__ constexpr int j0(int a, int b) { return a * 4 + b; }

__device__ inline void lds_barrier()
{
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();
}

// ---------------------------------------------------------------------------
// One KV segment (prefix or extend).
// ---------------------------------------------------------------------------
__device__ void accumulate_segment(const pa_fp8_kargs& kargs,
                                   const void* kv_base,
                                   const void* kv_rope_base,
                                   int kv_rows,
                                   const int* kv_indices,
                                   int sgl_page_shift,
                                   int sgl_rows_per_page,
                                   int sgl_scale_off,
                                   int page_begin,
                                   int valid_kv_len,
                                   char* smem,
                                   const i32x8 (&v_q)[pa_fp8_traits::Q_SUB]
                                                     [pa_fp8_traits::D_SLICES],
                                   const bf16x8 (&v_qr)[pa_fp8_traits::Q_SUB]
                                                       [pa_fp8_traits::ROPE_KST],
                                   const int (&q_sb)[pa_fp8_traits::Q_SUB],
                                   f32x4 (&v_o)[pa_fp8_traits::Q_SUB][pa_fp8_traits::O_LEN],
                                   float c_row,   // softmax_scale * log2e
                                   float (&m_row)[pa_fp8_traits::Q_SUB],
                                   float (&l_row)[pa_fp8_traits::Q_SUB],
                                   float (&m_ref)[pa_fp8_traits::Q_SUB],
                                   const float (&sink_log2)[pa_fp8_traits::Q_SUB])
{
    using T = pa_fp8_traits;

    const int max_e = *kargs.max_e_ptr;   // uniform; one s_load, hoisted
    const int num_tiles = pa_fp8_ceil_div(valid_kv_len, T::KV_TILE);
    if (num_tiles <= 0) return;

    const int tid = (int)threadIdx.x;
    int lane = tid % T::WARP_SIZE;
    asm volatile("" : "+v"(lane));  // break CSE with the Q-load decomposition
    const int warp = __builtin_amdgcn_readfirstlane(tid / T::WARP_SIZE);
    const int c    = lane & 15;
    const int g    = lane >> 4;

    auto g_kv = make_gmem(reinterpret_cast<const fp8_t*>(kv_base),
                          (unsigned)kv_rows * (unsigned)kargs.stride_kv_row);
    auto g_idx = make_gmem(kv_indices + page_begin, (unsigned)valid_kv_len * 4u);

    // RoPE tile first, scales behind it.  The RoPE tile is read with
    // ds_read_b64_tr_b8, whose 8x16 B transpose block is selected by *absolute*
    // LDS address bits, so smem_rope must stay aligned to ROW_R (512).
    // 2*TILE_BYTES is; putting the 256-byte scale array in front of it shifts
    // it half a row and every transposed RoPE read comes back scrambled -- NaN
    // for every non-empty case, with mode=empty still passing because it is a
    // no-op.  Total LDS is unchanged either way.  Keep anything new behind the
    // RoPE tile, or pad it to a multiple of ROW_R.
    char* const smem_rope = smem + 2 * T::TILE_BYTES;
    char* const smem_sc0  = smem_rope + T::ROPE_BUFS * T::ROPE_BYTES;
    // byte view: see the RoPE staging note below
    auto g_kvr = make_gmem(reinterpret_cast<const fp8_t*>(kv_rope_base),
                           (unsigned)kv_rows * (unsigned)kargs.stride_kvr_row * 2u);
    const __SIZE_TYPE__ kvr_row_bytes =
        (__SIZE_TYPE__)kargs.stride_kvr_row * 2u;

    // sglang keeps token ids in its own space (the top-k indexer produces
    // them), so the page->grid-row remap happens here, three ALU ops at the
    // point of use, rather than by rewriting the index kernel.
    const int sgl_mask = (1 << sgl_page_shift) - 1;
    // A *row index* fits 32 bits (4 G rows = 2.3 TB of 576-byte rows); only the
    // byte offset needs 64.  Written as a 64-bit expression the compiler has to
    // emit a full 64x32 multiply -- two v_mul_lo_u32 (quarter rate), a
    // v_mad_u64_u32 and a v_add3 -- for every slot, every tile.  Keeping the row
    // index in 32 bits and widening once leaves a single v_mad_u64_u32.
    // The 64-bit product itself is unchanged, so this is bit-identical; only the
    // intermediate is narrowed.
    auto page_row32 = [&](int gid) -> unsigned {
        return (unsigned)(gid >> sgl_page_shift) * (unsigned)sgl_rows_per_page;
    };
    auto grid_row32 = [&](int gid) -> unsigned {
        if constexpr (!T::SGL_PAGED) return (unsigned)gid;
        else return page_row32(gid) + (unsigned)(gid & sgl_mask);
    };
    auto grid_row = [&](int gid) -> __SIZE_TYPE__ {
#if (PA_ADDR32 & 1)
        return (__SIZE_TYPE__)grid_row32(gid);
#else
        if constexpr (!T::SGL_PAGED) return (__SIZE_TYPE__)gid;
        else return (__SIZE_TYPE__)(gid >> sgl_page_shift)
                        * (__SIZE_TYPE__)sgl_rows_per_page
                    + (__SIZE_TYPE__)(gid & sgl_mask);
#endif
    };
    // Byte offset of a token's E8M0 slot inside its page's scale region.  After
    // the collapse pass all seven per-64 exponents are equal, so byte 0 is the
    // token exponent.
    auto exp_off = [&](int gid) -> __SIZE_TYPE__ {
#if (PA_ADDR32 & 2)
        // Same narrowing.  The old form chained *two* 64-bit multiplies:
        // (u64)page * (u64)rows_per_page * (u64)stride.
        return (__SIZE_TYPE__)page_row32(gid) * (__SIZE_TYPE__)kargs.stride_kv_row
             + (__SIZE_TYPE__)sgl_scale_off
             + (__SIZE_TYPE__)(unsigned)((gid & sgl_mask) * 8);
#else
        return (__SIZE_TYPE__)(gid >> sgl_page_shift)
                   * (__SIZE_TYPE__)sgl_rows_per_page
                   * (__SIZE_TYPE__)kargs.stride_kv_row
             + (__SIZE_TYPE__)sgl_scale_off
             + (__SIZE_TYPE__)(gid & sgl_mask) * 8u;
#endif
    };

    // ---- per-lane LDS read bases ----------------------------------------
    // QK placement: element j of lane (c,g) carries
    //     d = 128*ds + 64*(j/16) + 16*g + (j%16)
    // i.e. two 16-byte cells at d_blk = 8*ds + g and 8*ds + 4 + g, so that one
    // hardware scale block coincides with one 32-wide E8M0 block.
    const int qk_base = (c >> 2) * T::ROW + (c & 3) * 16 + g * T::ATOM;
    // PV: slot block s_hi = 4*beta + g, atom = o_tile, lane supplies base + c*8
    const int pv_base = g * T::ROW + c * 8;
    // RoPE QK: token 16*nt+c -> row 4*nt + c/4, slot c%4; d = 32*st + 8*g + [0,8)
    const int qkr_base = (c >> 2) * T::ROW_R + (c & 3) * 32 + (g & 1) * 16;
    // RoPE PV: row 8*s + 4*h + g, lane supplies base + c*8
    const int pvr_base = g * T::ROW_R + c * 8;
    // RoPE staging.  opus's _async_load dispatches on the *element* count and
    // only implements 1/2/4/12/16, so a bf16 view caps out at 4 B/lane (count 2)
    // -- count 8 falls off the end of the if-constexpr chain and silently emits
    // nothing.  Viewing the same bytes as fp8 gets the full 16 B/lane
    // dwordx4-to-LDS, which is 4x fewer copy instructions.
    //
    // One copy writes 1024 contiguous bytes = two slot blocks.  Lane l lands at
    // byte 16l, i.e. slot block l/32, chunk (l%32)/8, token slot (l%8)/2, and
    // the low or high 8 bf16 of that chunk.
    const int rr_slot = (lane & 7) >> 1;                       // token slot in the row
    const int rr_doff = 32 * ((lane & 31) >> 3) + 16 * (lane & 1);  // byte offset in row

    // ---- async-copy plumbing --------------------------------------------
    // buffer_load_dwordx4-to-LDS writes lane l at (lds_base + l*16), which is
    // exactly cell (s_lo = l%8, d_blk = d_blk_base + l/8).  Wave w owns slot
    // blocks s_hi in [4w, 4w+4).
    const int p_lane = 16 * ((lane & 7) >> 2) + (lane & 3);  // token part from lane
    const int d_lane = (lane >> 3) * 16;                     // byte part from lane

    // Index / scale software pipeline.
    //
    // Run row indices two tiles ahead; commit gathered scales at tile tail.
    auto load_rows = [&](int tile_idx, int (&row)[T::SLOTS_PER_WAVE]) {
#pragma unroll
        for (int half = 0; half < T::SLOTS_PER_WAVE; ++half) {
            const int s_hi = T::SLOTS_PER_WAVE * warp + half;
            const int p    = 32 * (s_hi >> 2) + 4 * (s_hi & 3) + p_lane;
            row[half] = load(g_idx, tile_idx * T::KV_TILE + p)[0];
        }
    };
    auto issue_copy_r = [&]<int H0, int H1>(const int (&row)[T::SLOTS_PER_WAVE],
                                            int buf) {
        char* const kv_dst = smem + buf * T::TILE_BYTES;
#pragma unroll
        for (int half = H0; half < H1; ++half) {
            const int s_hi = T::SLOTS_PER_WAVE * warp + half;
            // unsigned, not int: a buffer resource addresses 4 GB and the
            // signed product overflowed at 2 GB, which 576-byte rows reach at
            // 3.7 M tokens/layer -- inside a real DSv4 pool.
            const __SIZE_TYPE__ goff =
                grid_row(row[half]) * (__SIZE_TYPE__)kargs.stride_kv_row
                + (__SIZE_TYPE__)d_lane;
#if PA_SGLANG_PAGED
            // sglang rows are 576 B: bytes 448..511 are RoPE bf16, not E8M0.
            // Nothing reads cells 28..31 here (the QK skips its tail slice,
            // stage_exps sources the exponent from the page scale region, and
            // 29/30 are overwritten anyway), so drop that quarter of the copy.
            // q=3's chunk is cells 24..31 and lanes 32..63 are the ones landing
            // in 28..31, so predicating them off copies exactly 448 B/token.
#pragma unroll
            for (int q = 0; q < 3; ++q)
                g_kv.template async_load_global<16>(
                    kv_dst + s_hi * T::ROW + q * 1024,
                    goff + (__SIZE_TYPE__)q * 128u);
            if (d_lane < 64)
                g_kv.template async_load_global<16>(
                    kv_dst + s_hi * T::ROW + 3 * 1024,
                    goff + (__SIZE_TYPE__)3 * 128u);
#else
#pragma unroll
            for (int q = 0; q < 4; ++q)
                g_kv.template async_load<16>(kv_dst + s_hi * T::ROW + q * 1024,
                                             goff + (unsigned)q * 128u);
#endif
        }
    };
    auto issue_copy = [&](const int (&row)[T::SLOTS_PER_WAVE], int buf) {
        issue_copy_r.template operator()<0, T::SLOTS_PER_WAVE>(row, buf);
    };
    // The 16 packed scale bytes of each token are gathered into their own LDS
    // array rather than read back out of the KV tile: the tile's tail gets
    // zeroed during requantisation, and an E8M0 byte of 127 is 0x7F, which is
    // NaN in e4m3 -- 0 * NaN would poison the MFMA even with Q's tail zeroed.
    const bool scale_lane = ((lane >> 3) == 0);
    using exp_pf_t = std::conditional_t<T::SGL_PAGED, i32x2, i32x4>;
    auto fetch_exps_r = [&]<int H0, int H1>(const int (&row)[T::SLOTS_PER_WAVE],
                                            exp_pf_t (&e)[T::SLOTS_PER_WAVE]) {
#pragma unroll
        for (int half = H0; half < H1; ++half) {
#if PA_SGLANG_PAGED
            // exp_off already points at the token's 8-byte slot, so one load
            // gets all seven per-64 exponents -- same instruction count as the
            // single byte it replaces.
            // A raw 64-bit pointer, NOT g_kv.  g_kv is a buffer resource, whose
            // num_records is 32 bits and is built from
            // `(unsigned)kv_rows * (unsigned)stride_kv_row` -- that product wraps
            // once the pool passes 4 GiB (a DSv4-Pro rank holds far more), and
            // every gather past the wrapped bound reads as *zero* rather than
            // faulting.  A zero exponent then makes the requant shift its block
            // out of range, so the row's KV is silently discarded.  The row copy
            // already avoids this by going through async_load_global; this
            // gather was the one place left on the buffer path.
            e[half] = scale_lane
                    ? *reinterpret_cast<const i32x2*>(
                          reinterpret_cast<const char*>(kv_base) + exp_off(row[half]))
                    : i32x2{0, 0};
#else
            e[half] = scale_lane
                    ? __builtin_bit_cast(i32x4, load<16>(g_kv, row[half] * kargs.stride_kv_row
                                                                + T::D_NOPE))
                    : i32x4{0, 0, 0, 0};
#endif
        }
    };
    auto fetch_exps = [&](const int (&row)[T::SLOTS_PER_WAVE],
                          exp_pf_t (&e)[T::SLOTS_PER_WAVE]) {
        fetch_exps_r.template operator()<0, T::SLOTS_PER_WAVE>(row, e);
    };
#if PA_EXP_GATHER_ASM && PA_SGLANG_PAGED
    // See PA_EXP_GATHER_ASM.  Four distinct address pairs, four loads back to
    // back, one drain -- instead of three serialised round trips.
    auto fetch_exps_pro = [&](const int (&row)[T::SLOTS_PER_WAVE],
                              exp_pf_t (&e)[T::SLOTS_PER_WAVE]) {
        static_assert(T::SLOTS_PER_WAVE == 4, "the asm block is written for four slots");
        const char* const b  = reinterpret_cast<const char*>(kv_base);
        const char* const p0 = b + exp_off(row[0]);
        const char* const p1 = b + exp_off(row[1]);
        const char* const p2 = b + exp_off(row[2]);
        const char* const p3 = b + exp_off(row[3]);
#pragma unroll
        for (int h = 0; h < T::SLOTS_PER_WAVE; ++h) e[h] = i32x2{0, 0};
        if (scale_lane) {
            asm volatile(
                "global_load_dwordx2 %0, %4, off\n\t"
                "global_load_dwordx2 %1, %5, off\n\t"
                "global_load_dwordx2 %2, %6, off\n\t"
                "global_load_dwordx2 %3, %7, off\n\t"
                // Inline VMEM outputs require an explicit drain before use.
                "s_waitcnt vmcnt(0)"
                : "=&v"(e[0]), "=&v"(e[1]), "=&v"(e[2]), "=&v"(e[3])
                : "v"(p0), "v"(p1), "v"(p2), "v"(p3)
                : "memory");
        }
    };
#endif
    auto commit_exps_r = [&]<int H0, int H1>(const exp_pf_t (&e)[T::SLOTS_PER_WAVE], int buf) {
        if (!scale_lane) return;
        char* dst = smem_sc0 + buf * T::SCALE_BYTES;
#pragma unroll
        for (int half = H0; half < H1; ++half) {
            const int s_hi = T::SLOTS_PER_WAVE * warp + half;
            const int p = 32 * (s_hi >> 2) + 4 * (s_hi & 3) + p_lane;
#if PA_SGLANG_PAGED
            *reinterpret_cast<i32x2*>(dst + p * 8) = e[half];
#else
            *reinterpret_cast<i32x4*>(dst + p * 16) = e[half];
#endif
        }
    };
    auto commit_exps = [&](const exp_pf_t (&e)[T::SLOTS_PER_WAVE], int buf) {
        commit_exps_r.template operator()<0, T::SLOTS_PER_WAVE>(e, buf);
    };

    // RoPE row indices for the *next* tile are fetched at the top of a tile and
    // consumed at its tail, so the gather latency is fully covered.
    auto load_rope_rows = [&](int tile_idx, int (&rr)[T::RCOPY]) {
#pragma unroll
        for (int i = 0; i < T::RCOPY; ++i)
            rr[i] = load(g_idx, tile_idx * T::KV_TILE
                                + (T::KV_TILE / T::NUM_WARPS) * warp + 8 * i
                                + 4 * (lane >> 5) + rr_slot)[0];
    };
    auto issue_rope_copy = [&](const int (&rr)[T::RCOPY], int rb) {
#pragma unroll
        for (int i = 0; i < T::RCOPY; ++i)
            g_kvr.template async_load_global<16>(
                smem_rope + rb * T::ROPE_BYTES
                    + (T::RROWS_PER_WAVE * warp + 2 * i) * T::ROW_R,
                grid_row(rr[i]) * kvr_row_bytes + (__SIZE_TYPE__)rr_doff);
    };

    // Build per-token exponent tables and normalize the staged NoPE tile.
#if PA_NO_COLLAPSE
    __shared__ int tile_max_e[PA_RQ_PIPELINE ? 2 : 1];
#endif
    auto stage_exps = [&](int b, int tbase) {
        char* const tp = smem + b * T::TILE_BYTES;
        auto kv   = make_smem(reinterpret_cast<fp8_t*>(tp));
        auto etok = make_smem(reinterpret_cast<u8_t*>(tp + T::ETOK_OFF));
        auto etkt = make_smem(reinterpret_cast<u8_t*>(tp + T::ETKT_OFF));
        const int tok = (T::RQ_SPLIT == 1) ? tid : (tid % T::KV_TILE);
        // rqh is wave-uniform (KV_TILE is a whole number of waves), so pin it
        // to an SGPR: it makes the half-predicate an s_cmp and keeps the VGPR
        // budget where it was.  This kernel is at 512/512 with zero scratch,
        // and leaving rqh in a VGPR costs 44 B of it.
        const int rqh = (T::RQ_SPLIT == 1) ? 0
                      : __builtin_amdgcn_readfirstlane(tid / T::KV_TILE);
        if (tok < T::KV_TILE && rqh < T::RQ_SPLIT) {
            const int shi = slot_hi_of(tok), slo = slot_lo_of(tok);
#if PA_SGLANG_PAGED
            // sglang's cell 28 holds RoPE bf16, not the exponent; it came from
            // the page's scale region via fetch/commit_exps instead.
            auto sx = make_smem(reinterpret_cast<u8_t*>(smem_sc0 + b * T::SCALE_BYTES));
            const auto sc8 = sx.template _load<8>(tok * 8);
            const u8_t* sbp = reinterpret_cast<const u8_t*>(&sc8);
            constexpr int SGL_NBLK = T::D_NOPE / 64;   // 7 block-64 over 448
            int et = 0;
#pragma unroll
            for (int blk = 0; blk < SGL_NBLK; ++blk)
                et = max(et, (int)(unsigned char)sbp[blk]);
#else
            int et = __builtin_bit_cast(
                int, kv.template _load<4>(shi * T::ROW + slo * 16
                                          + T::O_TILES * T::ATOM)) & 0xff;
#endif
#if PA_LDS_REQUANT
            // A token's 448 bytes are NOT contiguous: ATOM a holds d in
            // [16a, 16a+16) for all eight tokens of the slot block, so the
            // token's data is 28 chunks of 16 B at a*ATOM + slo*16.  A scale
            // block is 4 ATOMs under the paged layout (block-64) and 2 under
            // the flat one (block-32).
            {
#if PA_SGLANG_PAGED
                const u8_t* sb = sbp;                 // 7 per-64, already staged
                constexpr int ATOMS_PER_BLK = 4;
                const int em = et;                    // max of the seven
#else
                const auto sc = kv.template _load<16>(shi * T::ROW + slo * 16
                                                      + T::O_TILES * T::ATOM);
                const u8_t* sb = reinterpret_cast<const u8_t*>(&sc);
                constexpr int ATOMS_PER_BLK = 2;
                int em = 0;
#pragma unroll
                for (int blk = 0; blk < T::NBLK; ++blk)
                    em = max(em, (int)(unsigned char)sb[blk]);
#endif
                char* const rq = smem + b * T::TILE_BYTES + shi * T::ROW
                                 + slo * 16;
#if PA_RQ_SKIP == 0
#pragma unroll
                for (int a = 0; a < T::O_TILES; a += T::RQ_BATCH) {
                    // Each of the RQ_SPLIT threads on this token owns one
                    // contiguous run of ATOMs.  Expressed as a predicate inside
                    // the original loop rather than as a loop over a runtime
                    // base, because `sb` is register-resident: a runtime index
                    // into it puts the whole array in scratch.  After unrolling
                    // this is a compile-time side against a wave-uniform rqh,
                    // so it is an s_cbranch and costs no divergence.
                    if (T::RQ_SPLIT > 1 && ((a >= T::RQ_ATOMS ? 1 : 0) != rqh))
                        continue;
                    if constexpr (T::RQ_BATCH == 1) {
                        // verbatim the unbatched body, so RQ_BATCH 1 is
                        // codegen-identical to the kernel this came from
                        const int k = em - (int)(unsigned char)sb[a / ATOMS_PER_BLK];
#if PA_RQ_BRANCH
                        if (!k) continue;
#endif
                        unsigned* w = reinterpret_cast<unsigned*>(
                            smem + b * T::TILE_BYTES + shi * T::ROW + a * T::ATOM
                            + slo * 16);
#pragma unroll
                        for (int d = 0; d < 4; ++d)
                            w[d] = PA_RQ_NOOP ? (w[d] ^ (unsigned)(k & 0))
                                 : PA_RQ_CVT  ? pa_fp8::shift_exp_dword_cvt(w[d], k)
                                              : pa_fp8::shift_exp_dword(w[d], k);
                    } else {
                        // Every read of the batch is issued before the drain, so
                        // one LDS round trip covers RQ_BATCH ATOMs, not one.
                        unsigned t[T::RQ_BATCH][4];
#pragma unroll
                        for (int j = 0; j < T::RQ_BATCH; ++j) {
                            const unsigned* w = reinterpret_cast<const unsigned*>(
                                rq + (a + j) * T::ATOM);
#pragma unroll
                            for (int d = 0; d < 4; ++d) t[j][d] = w[d];
                        }
                        // The reads are compiler-emitted, so nothing stops the
                        // scheduler sinking each one to just above its own use --
                        // the serial round trip this batch exists to avoid.
                        // Laundering every dword in one place makes the whole
                        // batch live at once.  Plain scalars, not a vector: an
                        // `"+v"` on opus's vector_t wrapper binds one register
                        // and silently drops the other three -- it compiles, and
                        // writes dword 0 into all four lanes of the quad.
#pragma unroll
                        for (int j = 0; j < T::RQ_BATCH; ++j)
#pragma unroll
                            for (int d = 0; d < 4; ++d)
#if PA_RQ_AGPR
                                asm volatile("" : "+a"(t[j][d]) ::);
#else
                                asm volatile("" : "+v"(t[j][d]) ::);
#endif
#pragma unroll
                        for (int j = 0; j < T::RQ_BATCH; ++j) {
                            const int k = em
                                - (int)(unsigned char)sb[(a + j) / ATOMS_PER_BLK];
#if PA_RQ_BRANCH
                            if (!k) continue;
#endif
                            unsigned* w = reinterpret_cast<unsigned*>(
                                rq + (a + j) * T::ATOM);
#pragma unroll
                            for (int d = 0; d < 4; ++d)
                                w[d] = PA_RQ_NOOP ? (t[j][d] ^ (unsigned)(k & 0))
                                     : PA_RQ_CVT  ? pa_fp8::shift_exp_dword_cvt(t[j][d], k)
                                                  : pa_fp8::shift_exp_dword(t[j][d], k);
#if PA_RQ_SERIAL
                            // The batch exists to get every ds_read issued before
                            // the drain -- NOT to interleave the arithmetic.  Left
                            // alone the scheduler software-pipelines the shifts of
                            // all RQ_BATCH atoms, doubling the live temporaries and
                            // evicting the long-lived values that cross this pass.
                            // Serialise the arithmetic so every atom reuses one
                            // register quad; the reads are already in flight.
                            __builtin_amdgcn_sched_barrier(0);
#endif
                        }
                    }
                }
#endif  // PA_RQ_SKIP
                et = em;          // the row now shares this exponent
            }
#endif
            // One writer per token: both halves derive the same `et`.
            if (rqh == 0) {
#if PA_NO_COLLAPSE
                // Only real tokens set the tile frame.  A partial tile's padding
                // slots read the index array out of bounds, gather row 0, and
                // would otherwise put row 0's exponent on every token in the tile.
                // 128 threads hitting one LDS address serialise in hardware.
                // Fold across g first -- two permlanes, no registers, no LDS --
                // so only the 16 lanes with g == 0 issue an atomic: 32 per tile
                // instead of 128.  Invalid slots contribute 0 rather than being
                // masked out of the atomic, which is the same thing once the
                // fold is doing the masking.
#if PA_RQ_FOLD
                const int etm = (tbase + tok < valid_kv_len) ? et : 0;
                // ... and then across the 16 lanes of the row, leaving one
                // atomic per wave: 2 per tile instead of 128.
                const int etg = pa_fp8::reduce_max_row16_i(
                                    pa_fp8::reduce_max_groups_i(etm));
                if ((lane & 15) == 15) atomicMax(&tile_max_e[PA_ME_SLOT(b)], etg);
#else
                if (tbase + tok < valid_kv_len) atomicMax(&tile_max_e[PA_ME_SLOT(b)], et);
#endif
#endif
                etok.template store<1>((u8_t)et, tok);
#if PA_ETOK_F32
                // u8 view with byte offsets on purpose: the smem helpers
                // dispatch on element count and silently emit nothing for a
                // count they do not implement, so stay on the sizes s_etok
                // already proves work.
                {
                    auto ef = make_smem(reinterpret_cast<u8_t*>(tp + T::ETOKF_OFF));
                    ef.template store<4>(__builtin_bit_cast(
                        decltype(ef.template _load<4>(0)),
                        (unsigned)et << 23), tok * 4);
                }
#endif
                etkt.template store<1>((u8_t)et, (tok & 15) * 8 + (tok >> 4));
            }
        }
    };

    int row_pf[T::SLOTS_PER_WAVE], row_pf2[T::SLOTS_PER_WAVE], rr_pf[T::RCOPY];
    exp_pf_t e_pf[T::SLOTS_PER_WAVE];
    load_rows(0, row_pf);
#if PA_SGLANG_PAGED
#if PA_EXP_GATHER_ASM
    fetch_exps_pro(row_pf, e_pf);
#else
    fetch_exps(row_pf, e_pf);
#endif
    commit_exps(e_pf, 0);
#endif
    issue_copy(row_pf, 0);
    load_rope_rows(0, rr_pf);
    issue_rope_copy(rr_pf, 0);
    if constexpr (T::ROPE_BUFS == 2) load_rope_rows(1, rr_pf);
    load_rows(1, row_pf);   // indices for tile 1, awaited by the first barrier
#if PA_RQ_PIPELINE
    // Tile 0 has no preceding PV to overlap with, so use a compact rolled
    // requantization path instead of instantiating stage_exps twice.
    static_assert(PA_SGLANG_PAGED, "the cold path reads the paged scale region");
    {
        if (tid == 0) tile_max_e[0] = 0;
        s_waitcnt_vmcnt(0_I);   // tile 0's copy, issued just above
        lds_barrier();          // ... and every other wave's slice of it
        if (tid < T::KV_TILE) {
            const int tok = tid;
            const int shi = slot_hi_of(tok), slo = slot_lo_of(tok);
            auto sx0 = make_smem(reinterpret_cast<u8_t*>(smem_sc0));
            const i32x2 scw = __builtin_bit_cast(
                i32x2, sx0.template _load<8>(tok * 8));
            // Read the block byte out of two registers rather than indexing a
            // register-resident array: a runtime index there lands in scratch.
            auto sbyte = [&](int blk) {
                return ((blk < 4 ? scw[0] : scw[1]) >> (8 * (blk & 3))) & 0xff;
            };
            int em = 0;
            for (int blk = 0; blk < T::D_NOPE / 64; ++blk) em = max(em, sbyte(blk));
            char* const rq = smem + shi * T::ROW + slo * 16;
#pragma unroll 1
            for (int a = 0; a < T::O_TILES; ++a) {
                const int k = em - sbyte(a / 4);
                unsigned* w = reinterpret_cast<unsigned*>(rq + a * T::ATOM);
#pragma unroll
                for (int d = 0; d < 4; ++d)
                    w[d] = PA_RQ_CVT ? pa_fp8::shift_exp_dword_cvt(w[d], k)
                                     : pa_fp8::shift_exp_dword(w[d], k);
            }
            const int etm = (tok < valid_kv_len) ? em : 0;
            const int etg = pa_fp8::reduce_max_row16_i(
                                pa_fp8::reduce_max_groups_i(etm));
            if ((lane & 15) == 15) atomicMax(&tile_max_e[0], etg);
            make_smem(reinterpret_cast<u8_t*>(smem + T::ETOK_OFF))
                .template store<1>((u8_t)em, tok);
            make_smem(reinterpret_cast<u8_t*>(smem + T::ETKT_OFF))
                .template store<1>((u8_t)em, (tok & 15) * 8 + (tok >> 4));
        }
    }
#endif
#if PA_ICACHE_PROBE
    {   // 只跑一次,和 prologue 那次重量化同量级
        int acc = tid;
#pragma unroll
        for (int i = 0; i < PA_ICACHE_PROBE; ++i) acc = acc * 3 + (acc >> 7) + i;
        if (acc == 0x7fffffff) smem[0] = (char)acc;
    }
#endif

    int buf = 0, rbuf = 0;
    for (int tile = 0; tile < num_tiles; ++tile) {
        char* const tilep = smem + buf * T::TILE_BYTES;
        auto s_kv   = make_smem(reinterpret_cast<fp8_t*>(tilep));
        auto s_exp  = make_smem(reinterpret_cast<u8_t*>(smem_sc0 + buf * T::SCALE_BYTES));
        auto s_etok = make_smem(reinterpret_cast<u8_t*>(tilep + T::ETOK_OFF));
        auto s_etkt = make_smem(reinterpret_cast<u8_t*>(tilep + T::ETKT_OFF));
#if PA_ETOK_F32
        auto s_etokf = make_smem(reinterpret_cast<u8_t*>(tilep + T::ETOKF_OFF));
#endif

        // One barrier per tile.  Reaching it means every wave has finished the
        // PV reads of the *other* buffer, so the next tile's copy can be issued
        // immediately after it — no second barrier needed.
        s_waitcnt_vmcnt(number<PA_TOP_VMCNT>{});
        lds_barrier();
#if PA_NO_COLLAPSE
        // Reset before any wave can update the next tile's maximum.
        if (tid == 0) tile_max_e[PA_ME_SLOT(buf ^ 1)] = 0;
        lds_barrier();
#endif

        const bool has_next = (tile + 1 < num_tiles);

#if PA_SGLANG_PAGED
#if PA_GATHER_MODE == 1
        // Every slot's load is issued up here; the ones beyond GATHER_SPLIT are
        // committed immediately below, so their e_pf dies before the requant
        // pass instead of living across it.
        if (has_next) fetch_exps(row_pf, e_pf);
#elif PA_GATHER_SPLIT > 0
        if (has_next)
            fetch_exps_r.template operator()<0, PA_GATHER_SPLIT>(row_pf, e_pf);
#endif
#endif
        // writes buf^1 while the staging pass below touches buf, so it can be
        // issued here and pick up the staging pass as extra cover
#if PA_COPY_SPLIT
        if (has_next)
            issue_copy_r.template operator()<0, PA_COPY_SPLIT>(row_pf, buf ^ 1);
#else
        if (has_next) issue_copy(row_pf, buf ^ 1);
#endif
        if constexpr (T::ROPE_BUFS == 2) {
            // double-buffered: no need to wait for every wave's PV-RoPE reads,
            // so the copy moves up here and gets a whole tile of cover
            if (has_next) issue_rope_copy(rr_pf, rbuf ^ 1);
        }

#if PA_SGLANG_PAGED && PA_GATHER_MODE == 1 && PA_GATHER_SPLIT < 4
        // kill the tail slots' e_pf here, after issue_copy has covered them
        if (has_next)
            commit_exps_r.template operator()<PA_GATHER_SPLIT, T::SLOTS_PER_WAVE>(
                e_pf, buf ^ 1);
#endif
        // ---- staging requantisation: per-32 E8M0 -> one per-token exponent
        // REQ_SPLIT threads per token, thread q handling blocks q, q+2, ... and
        // zeroing tail cells 28+q, 30+q.  Each thread touches only its own
        // token, so no barrier is needed inside this pass.
#if !PA_RQ_PIPELINE
        stage_exps(buf, tile * T::KV_TILE);
#endif
        // Align all waves before QK consumes the staged tile.
        lds_barrier();
#if PA_COPY_SPLIT
        // second half of the staging copy: still a full tile ahead of its own
        // compute, but no longer back to back with the first half
        if (has_next)
            issue_copy_r.template operator()<PA_COPY_SPLIT, T::SLOTS_PER_WAVE>(
                row_pf, buf ^ 1);
#endif
#if PA_NO_COLLAPSE
        const int max_e_t = tile_max_e[PA_ME_SLOT(buf)];
        const int pv_e0   = max_e_t;   // byte = 127 + (max_e_t + dexp - MAXE), MAXE = 127
#else
        const int max_e_t = max_e;
        const int pv_e0   = 127;
#endif

        if (has_next) {
            // The gather is issued above the requantisation pass and committed
            // below it, so the gather overlaps requantization and e_pf dies
            // before QK.
#if PA_SGLANG_PAGED && PA_GATHER_MODE == 1
            commit_exps_r.template operator()<0, PA_GATHER_SPLIT>(e_pf, buf ^ 1);
#elif PA_SGLANG_PAGED && PA_GATHER_SPLIT < 4
            // The slots not issued above the requant pass.  Every slot left up
            // there keeps its e_pf live across the whole pass.  Split the
            // remaining slots to bound register lifetime.
            fetch_exps_r.template operator()<PA_GATHER_SPLIT, T::SLOTS_PER_WAVE>(
                row_pf, e_pf);
#endif
#if PA_SGLANG_PAGED && PA_GATHER_MODE != 1
            commit_exps(e_pf, buf ^ 1);       // buf^1 was last read by tile-1
#endif
            load_rows(tile + 2, row_pf2);     // OOB tiles read 0 from the buffer rsrc
            // double-buffered, the copy is issued at the *top* of the next
            // tile, so the indices must run one tile further ahead
#if !PA_ROPE_LATE
            load_rope_rows(tile + (T::ROPE_BUFS == 2 ? 2 : 1), rr_pf);
#endif
        }

        // v_p and dexp are produced by the softmax and consumed by the PV, which
        // the pipelined requant now sits between, so they live outside both.
        i32x8 v_p[T::Q_SUB];
        int   dexp[T::Q_SUB];   // new_m - m_ref, an exact non-negative integer
        // ---- QK: S[Q_SUB][16 heads, 128 tokens] --------------------------
        // the eight per-token exponents this lane needs as scale_a, one per
        // n-subtile, laid out so a single ds_read_b64 fetches all of them
        const i32x2 etk = __builtin_bit_cast(i32x2, s_etkt.template _load<8>(c * 8));
        auto s_rope = make_smem(reinterpret_cast<bf16_t*>(smem_rope + rbuf * T::ROPE_BYTES));

        float v_s[T::Q_SUB][T::S_LEN];
        // The K operands of n-subtile nt+1 are issued before nt's mfmas, so the
        // LDS round trip hides behind 8 mfmas instead of stalling in front of
        // them.  These are plain _loads, so the compiler owns their waitcnt --
        // unlike the PV's tr_loads, there is no partial-lgkmcnt hazard here.
        auto load_k_slice = [&](auto i_nt, auto i_ds) {
            constexpr int nt = decltype(i_nt)::value;
            constexpr int ds = decltype(i_ds)::value;
            constexpr int nt_off = (nt >> 1) * 4 * T::ROW + (nt & 1) * 64;
            constexpr int off0 = nt_off + ds * 1024;
            // The hi half of the last slice is d = 448..511: the packed E8M0
            // bytes and the pad, never real data.  Q's matching half is
            // already zero, so feed a zero register instead of reading the
            // packed scale/padding cells.
            const i32x4 hi = (ds == T::D_SLICES - 1)
                ? i32x4{0, 0, 0, 0}
                : __builtin_bit_cast(i32x4,
                      s_kv.template _load<16>(qk_base + off0 + 4 * T::ATOM));
            return pack32(
                __builtin_bit_cast(i32x4,
                    s_kv.template _load<16>(qk_base + off0)), hi);
        };
        auto load_k = [&](auto i_nt, i32x8 (&kk)[T::D_SLICES]) {
            static_for<T::D_SLICES>([&](auto i_ds) {
                kk[i_ds.value] = load_k_slice(i_nt, i_ds);
            });
        };
        auto load_kr = [&](auto i_nt, bf16x8 (&kr)[T::ROPE_KST]) {
            constexpr int nt = decltype(i_nt)::value;
            static_for<T::ROPE_KST>([&](auto i_st) {
                constexpr int st = i_st.value;
                kr[st] = __builtin_bit_cast(bf16x8, s_rope.template _load<8>(
                    qkr_base + nt * 4 * T::ROW_R + st * 256 + (g >> 1) * 128));
            });
        };

#if PA_QK_PREFETCH_K >= 0
        constexpr int KPF = PA_QK_PREFETCH_K;
        static_assert(KPF >= 0 && KPF <= T::D_SLICES, "invalid partial K prefetch");
        i32x8 kpf[2][KPF > 0 ? KPF : 1];
        i32x8 kcur[T::D_SLICES];
        static_for<T::D_SLICES>([&](auto i_ds) {
            constexpr int ds = i_ds.value;
            if constexpr (ds < KPF)
                kpf[0][ds] = load_k_slice(number<0>{}, i_ds);
            else
                kcur[ds] = load_k_slice(number<0>{}, i_ds);
        });
#else
        i32x8  kkb[T::QK_PIPE_K][T::D_SLICES];
        load_k(number<0>{}, kkb[0]);
#endif
        bf16x8 krb[T::QK_PIPE_R][T::ROPE_KST];
        load_kr(number<0>{}, krb[0]);
        static_for<T::N_TILES>([&](auto i_nt) {
            constexpr int nt = i_nt.value;
#if PA_QK_PREFETCH_K >= 0
            if constexpr (nt + 1 < T::N_TILES) {
                static_for<KPF>([&](auto i_ds) {
                    kpf[(nt + 1) & 1][i_ds.value] =
                        load_k_slice(number<nt + 1>{}, i_ds);
                });
            }
            if constexpr (nt > 0) {
                static_for<T::D_SLICES>([&](auto i_ds) {
                    constexpr int ds = i_ds.value;
                    if constexpr (ds >= KPF)
                        kcur[ds] = load_k_slice(number<nt>{}, i_ds);
                });
            }
#else
            if constexpr (T::QK_PIPE_K == 2) {
                if constexpr (nt + 1 < T::N_TILES) {
                    load_k(number<nt + 1>{}, kkb[(nt + 1) & 1]);
                }
            } else if constexpr (nt > 0) {
                load_k(number<nt>{}, kkb[0]);
            }
#endif
            if constexpr (T::QK_PIPE_R == 2) {
                if constexpr (nt + 1 < T::N_TILES) {
                    load_kr(number<nt + 1>{}, krb[(nt + 1) & 1]);
                }
            } else if constexpr (nt > 0) {
                load_kr(number<nt>{}, krb[0]);
            }
            const int sa = (etk[nt >> 2] >> (8 * (nt & 3))) & 0xff;
            f32x4 acc[T::Q_SUB];
#pragma unroll
            for (int qs = 0; qs < T::Q_SUB; ++qs) acc[qs] = f32x4{0.f, 0.f, 0.f, 0.f};
            // one K operand feeds both mfma tiles -- this is where the halved
            // LDS read traffic comes from
            static_for<T::D_SLICES>([&](auto i_ds) {
                constexpr int ds = i_ds.value;
                static_for<T::Q_SUB>([&](auto i_qs) {
                    constexpr int qs = i_qs.value;
#if PA_MFMA_NOP
                    // Optional separator between independent MFMA chains.
                    if constexpr (qs > 0) asm volatile("s_nop 0");
#endif
#if PA_QK_PREFETCH_K >= 0
                    if constexpr (ds < KPF)
                        acc[qs] = mma_qk<ds>(
                            kpf[nt & 1][ds], v_q[qs][ds],
                            acc[qs], sa, q_sb[qs]);
                    else
                        acc[qs] = mma_qk<ds>(
                            kcur[ds], v_q[qs][ds],
                            acc[qs], sa, q_sb[qs]);
#else
                    acc[qs] = mma_qk<ds>(
                            kkb[(T::QK_PIPE_K == 2) ? (nt & 1) : 0][ds],
                            v_q[qs][ds], acc[qs], sa, q_sb[qs]);
#endif
                });
            });
            // RoPE contributes to the very same S tile (same C layout)
            static_for<T::ROPE_KST>([&](auto i_st) {
                constexpr int st = i_st.value;
#if PA_MFMA_ASM
                // Keep the pair and its separator in one fixed asm group.
                static_assert(T::Q_SUB == 2, "hand-scheduled pair assumes Q_SUB==2");
                {
                    f32x4 d0, d1;
                    mma_bf16_pair(d0, d1, acc[0], acc[1],
                                  krb[(T::QK_PIPE_R == 2) ? (nt & 1) : 0][st],
                                  v_qr[0][st], v_qr[1][st]);
                    acc[0] = d0; acc[1] = d1;
                }
#else
                static_for<T::Q_SUB>([&](auto i_qs) {
                    constexpr int qs = i_qs.value;
#if PA_MFMA_NOP
                    if constexpr (qs > 0) asm volatile("s_nop 0");
#endif
                    acc[qs] = mma_bf16(
                        krb[(T::QK_PIPE_R == 2) ? (nt & 1) : 0][st],
                        v_qr[qs][st], acc[qs]);
                });
#endif
            });
#pragma unroll
            for (int qs = 0; qs < T::Q_SUB; ++qs)
#pragma unroll
                for (int i = 0; i < 4; ++i) v_s[qs][nt * 4 + i] = acc[qs][i];
            // Keep DS reads and MFMA issues grouped in the subtile's natural
            // ratio.  0x100 selects DS reads and 0x008 selects MFMA.
            static_for<PA_SCHED_REP>([&](auto) {
#if PA_SCHED_MODE == 0
                __builtin_amdgcn_sched_group_barrier(0x100, PA_SCHED_DS, 0);
                __builtin_amdgcn_sched_group_barrier(0x008, PA_SCHED_MFMA, 0);
#elif PA_SCHED_MODE == 1
                // Interleave one VALU between MFMA issues.
                __builtin_amdgcn_sched_group_barrier(0x100, PA_SCHED_DS, 0);
                static_for<PA_SCHED_MFMA>([&](auto) {
                    __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
                    __builtin_amdgcn_sched_group_barrier(0x002, 1, 0);
                });
#elif PA_SCHED_MODE == 2
                // Allow any ALU as the MFMA separator.
                __builtin_amdgcn_sched_group_barrier(0x100, PA_SCHED_DS, 0);
                static_for<PA_SCHED_MFMA>([&](auto) {
                    __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
                    __builtin_amdgcn_sched_group_barrier(0x001, 1, 0);
                });
#elif PA_SCHED_MODE == 3
                // Place two VALU fillers per MFMA.
                __builtin_amdgcn_sched_group_barrier(0x100, PA_SCHED_DS, 0);
                static_for<PA_SCHED_MFMA>([&](auto) {
                    __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
                    __builtin_amdgcn_sched_group_barrier(0x002, 2, 0);
                });
#endif
            });
        });

        // ---- per-token dequant, mask, online softmax ---------------------
        const bool last = (tile == num_tiles - 1);
        const int tile_base = tile * T::KV_TILE;
        float bias[T::Q_SUB], rsum[T::Q_SUB];
#if PA_P_SCALE_MUL
        float pscale[T::Q_SUB];
#endif


        // num_tiles is block-uniform, so this is a scalar branch: the 64
        // compare/select pairs run on the final tile only instead of all eight.
        if (last) {
#pragma unroll
            for (int qs = 0; qs < T::Q_SUB; ++qs)
#pragma unroll
                for (int a = 0; a < T::N_TILES; ++a)
#pragma unroll
                    for (int b = 0; b < 4; ++b)
                        if (tile_base + 16 * a + 4 * g + b >= valid_kv_len)
                            v_s[qs][a * 4 + b] = -1e30f;
        }

        // phase 1: row max.  Independent per sub-tile.  v_s is already fully
        // dequantised: both operand scales were applied by the MFMA itself.
        static_for<T::Q_SUB>([&](auto i_qs) {
            constexpr int qs = i_qs.value;
            float rmax[T::N_TILES];
#pragma unroll
            for (int a = 0; a < T::N_TILES; ++a) {
                // pairwise, so the 32 maxima form a depth-5 tree instead of a
                // 32-long dependent chain
                rmax[a] = max(max(v_s[qs][j0(a, 0)], v_s[qs][j0(a, 1)]),
                              max(v_s[qs][j0(a, 2)], v_s[qs][j0(a, 3)]));
            }
#pragma unroll
            for (int w = T::N_TILES / 2; w > 0; w >>= 1)
#pragma unroll
                for (int a = 0; a < w; ++a) rmax[a] = max(rmax[a], rmax[a + w]);
            const float row_max = reduce_max_groups(rmax[0]) * c_row;  // log2 domain

            // Keep the accumulator in the first tile's integer log2 frame.
            // The per-tile power of two is carried by PV scale_b.  Floor keeps
            // the correction exactly representable and the FP8 P operand bounded.
            const float new_m = __builtin_floorf(max(m_row[qs], row_max));
            if (m_ref[qs] == opus::numeric_limits<float>::lowest()) {
                m_ref[qs] = new_m;
                // Add the sink when the fixed frame is initialized.
                l_row[qs] = __builtin_amdgcn_exp2f(min(sink_log2[qs] - new_m, 96.f));
            }
            m_row[qs] = new_m;
            // exp2 is taken straight into the m_ref frame, so praw already
            // carries 2^dexp and the bf16 RoPE copy, rsum and l_row need no
            // correction of their own; only the fp8 operand has to undo it,
            // which folds into the per-token ldexp it was doing anyway.
            bias[qs]  = -m_ref[qs];
            dexp[qs]  = (int)(new_m - m_ref[qs]);
            rsum[qs]  = 0.f;
#if PA_P_SCALE_MUL
            // v_cvt_scalef32_pk_fp8_f32 divides by this power-of-two scale.
            // Its exponent field is max_e_t + dexp - 7.
            // max_e_t is a real E8M0 byte (~120) and dexp is bounded by the fixed
            // frame's ~96 octaves, so the field sits well inside [1,254]; the
            // clamp only covers an all-padding tile, where max_e_t is 0 and every
            // praw is 0 anyway.
            pscale[qs] = __builtin_bit_cast(
                float, (unsigned)max(max_e_t + dexp[qs] - 7, 1) << 23);
#endif
        });

        // phase 2: exp2 -> fp8 P, fused with the bf16 PV-RoPE.
        //
        // exp2(S*c_row - m) is one FMA plus one transcendental, then straight
        // into fp8: element j = 4a+b lands in byte b of dword a, which is exactly
        // the MFMA B-operand order.  P8 = fp8(P * 2^(e_tok - max_e + 7)); the
        // exponent is an integer so a single v_ldexp_f32 does the whole
        // per-token dequant-and-gain.
        //
        // V_rope is plain bf16, so its PV needs the *unscaled* P.  Holding that
        // bf16 copy for all 128 tokens costs 32 ArchVGPRs; emitting it one
        // 32-token k-step at a time and consuming it immediately costs 8.  The
        // rope V reads and the e_tok read are shared by both sub-tiles.
        s_waitcnt_lgkmcnt(0_I);  // lgkmcnt counts LDS stores; drain before tr_loads
        static_for<T::PV_R_KST>([&](auto i_st) {
            constexpr int st = i_st.value;
            bf16x8 pr[T::Q_SUB];
            // Issue RoPE V reads before computing this k-step's P.
            vector_t<bf16_t, 4> r[2 * T::O_TILES_R];
            static_for<T::O_TILES_R>([&](auto i_otr) {
                constexpr int otr = i_otr.value;
                r[2 * otr]     = __builtin_bit_cast(vector_t<bf16_t, 4>,
                    s_rope.template _tr_load<4, (8 * st) * T::ROW_R + otr * 128>(pvr_base));
                r[2 * otr + 1] = __builtin_bit_cast(vector_t<bf16_t, 4>,
                    s_rope.template _tr_load<4, (8 * st + 4) * T::ROW_R + otr * 128>(pvr_base));
            });
#pragma unroll
            for (int aa = 0; aa < 2; ++aa) {
                const int a = 2 * st + aa;
#if !PA_ETOK_F32
                const int ew = __builtin_bit_cast(
                    int, s_etok.template _load<4>(16 * a + 4 * g));
#endif
#if PA_P_SCALE_MUL
                // The four scales depend only on the token, not on the query
                // sub-tile, so they are built once here rather than twice
                // inside the Q_SUB loop, and as a vector so the apply becomes
                // two v_pk_mul_f32 instead of four v_mul_f32.
#if PA_ETOK_F32
                const f32x4 sc4 = __builtin_bit_cast(
                    f32x4, s_etokf.template _load<16>((16 * a + 4 * g) * 4));
#else
                f32x4 sc4;
#pragma unroll
                for (int b = 0; b < 4; ++b)
                    sc4[b] = __builtin_bit_cast(
                        float, (unsigned)((ew >> (8 * b)) & 0xff) << 23);
#endif
#endif
                static_for<T::Q_SUB>([&](auto i_qs) {
                    constexpr int qs = i_qs.value;
                    vector_t<float, 4> p4, praw;
#pragma unroll
                    for (int b = 0; b < 4; ++b) {
                        const float p = __builtin_amdgcn_exp2f(
                            __builtin_fmaf(v_s[qs][a * 4 + b], c_row, bias[qs]));
                        praw[b] = p;
#if !PA_P_SCALE_MUL
                        p4[b] = __builtin_ldexpf(
                            p, ((ew >> (8 * b)) & 0xff) - max_e_t + 7 - dexp[qs]);
#endif
                    }
#if PA_P_SCALE_MUL
                    // The E8M0 byte is an FP32 exponent field.  Apply its exact
                    // power of two element-wise; the pack carries the remaining
                    // tile-uniform scale.
#pragma unroll
                    for (int b = 0; b < 4; ++b) p4[b] = praw[b] * sc4[b];
#endif
                    rsum[qs] += (praw[0] + praw[1]) + (praw[2] + praw[3]);
#if PA_P_SCALE_MUL
                    const int hw = pa_fp8::pack_fp8_scaled(p4[0], p4[1], p4[2],
                                                           p4[3], pscale[qs]);
#else
                    const int hw = __builtin_bit_cast(int, cast<fp8_t>(p4));
#endif
                    v_p[qs][a] = hw;
#pragma unroll
                    for (int b = 0; b < 4; ++b) pr[qs][aa * 4 + b] = (bf16_t)praw[b];
                });
            }
            // The wait carries no data dependency on the asm outputs, so the
            // moves that assemble vr from them are free to float above it -- one
            // launder after the wait pins them down.  A tr_load feeding an mfma
            // operand directly escapes this, which is why the NoPE PV does not
            // need it.
            s_waitcnt_lgkmcnt(0_I);
            asm volatile("" : "+v"(r[0]), "+v"(r[1]), "+v"(r[2]), "+v"(r[3]),
                              "+v"(r[4]), "+v"(r[5]), "+v"(r[6]), "+v"(r[7]) ::);
            static_for<T::O_TILES_R>([&](auto i_otr) {
                constexpr int otr = i_otr.value;
                constexpr int o0  = T::O_TILES + otr;
                bf16x8 vr;
#pragma unroll
                for (int t = 0; t < 4; ++t) {
                    vr[t]     = r[2 * otr][t];
                    vr[4 + t] = r[2 * otr + 1][t];
                }
                static_for<T::Q_SUB>([&](auto i_qs) {
                    constexpr int qs = i_qs.value;
                    v_o[qs][o0] = mma_bf16_acc(vr, pr[qs], v_o[qs][o0]);
                });
            });
        });
#pragma unroll
        for (int qs = 0; qs < T::Q_SUB; ++qs)
            l_row[qs] += reduce_sum_groups(rsum[qs]);

        // ---- PV: O[Q_SUB][16 heads, 448] += P @ V ------------------------
        // ds_read_b64_tr_b8 is emitted from inline asm, so the compiler cannot
        // insert its waitcnt: every tr_load must be covered by an explicit
        // s_waitcnt lgkmcnt.  One o_tile is kept in flight ahead of the MFMA.
        // _tr_load's second template argument is the hardware `offset:` field.
        // Everything here is a compile-time byte count (max 27*128 + 3*4*4160 =
        // 53376, well inside the 16-bit field), so all 112 reads share ONE
        // runtime address register.  Left in the runtime operand they cost a
        // v_add each -- 126 of the kernel's 432 literal address adds -- plus a
        // live register per read in flight.
        auto load_v = [&](auto i_ot) {
            constexpr int ot = decltype(i_ot)::value;
            i32x8 v;
            static_for<T::K_BLOCKS>([&](auto i_beta) {
                constexpr int beta = i_beta.value;
                auto r = __builtin_bit_cast(
                    i32x2, s_kv.template _tr_load<8, ot * T::ATOM + beta * 4 * T::ROW>(
                               pv_base));
                v[beta * 2 + 0] = r[0];
                v[beta * 2 + 1] = r[1];
            });
            return v;
        };

        // Partial LGKM counts below assume a queue containing only PV tr_loads.
        // Drain compiler-managed LDS operations before entering that region.
#if PA_RQ_PIPELINE
        // vmcnt(0) covers this wave's own slice of the tile+1 copy; the barrier
        // makes it collective, because stage_exps touches every token while a
        // wave only copied its own four slot blocks.  That barrier also sits
        // after the RoPE PV, so it already means "every wave has finished
        // reading the rope buffer" -- which is the tail barrier's only job, and
        // why that one goes instead of this one being an addition.
        s_waitcnt_vmcnt(0_I);
        lds_barrier();
        if (has_next) stage_exps(buf ^ 1, (tile + 1) * T::KV_TILE);
#endif
        asm volatile("" ::: "memory");
        s_waitcnt_lgkmcnt(0_I);

        // Fetch V in PV_BATCH groups, then consume each result under explicit
        // partial LGKM waits.
        static_for<T::O_TILES / T::PV_BATCH>([&](auto i_b) {
            constexpr int b0 = i_b.value * T::PV_BATCH;
            i32x8 vv[T::PV_BATCH];
            static_for<T::PV_BATCH>([&](auto i_j) {
                constexpr int j = i_j.value;
                vv[j] = load_v(number<b0 + j>{});
            });
#if PA_PV_PARTIAL_WAIT
            // LDS retires in order, so each MFMA waits only for its vv[j].
            // The queue must contain no operation outside this tr_load batch.
            static_for<T::PV_BATCH>([&](auto i_j) {
                constexpr int j = i_j.value;
                s_waitcnt_lgkmcnt(number<T::PV_BATCH - 1 - j>{});
                asm volatile("" : "+v"(vv[j]) ::);
                static_for<T::Q_SUB>([&](auto i_qs) {
                    constexpr int qs = i_qs.value;
                    v_o[qs][b0 + j] =
                        mma_sb_acc(vv[j], v_p[qs], v_o[qs][b0 + j],
                                   pv_e0 + dexp[qs]);
                });
            });
#else
            s_waitcnt_lgkmcnt(0_I);
            // s_waitcnt carries no data dependency on the asm outputs.  While
            // the batch is small the compiler coalesces each tr_load result
            // straight into its mfma operand and there is nothing to reorder;
            // once register pressure forces real v_movs, those float above the
            // wait and the mfma reads pre-load garbage.  Launder to make the
            // dependency real.  Same trap as the PV-RoPE vr assembly.
#pragma unroll
            for (int j = 0; j < T::PV_BATCH; ++j) asm volatile("" : "+v"(vv[j]) ::);
            static_for<T::PV_BATCH>([&](auto i_j) {
                constexpr int j = i_j.value;
                static_for<T::Q_SUB>([&](auto i_qs) {
                    constexpr int qs = i_qs.value;
                    v_o[qs][b0 + j] =
                        mma_sb_acc(vv[j], v_p[qs], v_o[qs][b0 + j],
                                   pv_e0 + dexp[qs]);
                });
            });
#endif
        });

        if (has_next) {
#pragma unroll
            for (int i = 0; i < T::SLOTS_PER_WAVE; ++i) row_pf[i] = row_pf2[i];

            // Refill the single RoPE buffer after all waves finish reading it.
            if constexpr (T::ROPE_BUFS == 1) {
                // single-buffered: it can only be refilled once every wave has
                // finished reading it
#if !PA_RQ_PIPELINE
                lds_barrier();
#endif
#if PA_ROPE_LATE
                // rr_pf otherwise lives from mid-tile across the whole QK/PV
                // span -- RCOPY(4) ArchVGPRs held through the mfma region for
                // no reason but to cover its own index load.  Same trade as
                // PA_GATHER_SPLIT: give up that cover, get the registers.
                load_rope_rows(tile + 1, rr_pf);
#endif
                issue_rope_copy(rr_pf, 0);
            }

        }
        buf ^= 1;
        if constexpr (T::ROPE_BUFS == 2) rbuf ^= 1;
    }
    // num_tiles is block-uniform, so this barrier is never divergent.  It keeps
    // the next segment from reusing LDS while a wave is still in the last PV.
    lds_barrier();
}

} // namespace pa_fp8

// ---------------------------------------------------------------------------
// 4 waves of 64 == one wave per SIMD, which is what lifts the VGPR budget to
// 512.  __launch_bounds__'s second argument is min-waves-per-EU; 1 is what tells
// the backend it may use the whole file.
__global__ __launch_bounds__(pa_fp8_traits::BLOCK_SIZE, 1)
void pa_sparse_mla_kernel(pa_fp8_kargs kargs)
{
    using namespace opus;
    using namespace pa_fp8;
    using T = pa_fp8_traits;

    // Default is reverse order so the heaviest rows dispatch first; row_map
    // overrides it wholesale (see the field's comment for why placement pays).
    const int q_token = kargs.row_map ? kargs.row_map[blockIdx.x]
                                      : kargs.N - 1 - (int)blockIdx.x;
    const int h_block = (int)blockIdx.y;

    const int tid  = (int)threadIdx.x;
    const int lane = tid % T::WARP_SIZE;
    const int warp = __builtin_amdgcn_readfirstlane(tid / T::WARP_SIZE);
    const int c    = lane & 15;
    const int g    = lane >> 4;

    const int h_base = h_block * T::HEADS_PER_BLOCK;
    // wave w owns heads [32w, 32w+32); sub-tile qs owns [32w+16qs, 32w+16qs+16)
    const int h_wave = warp * (T::Q_SUB * T::Q_TILE);

    __shared__ char smem[T::smem_size_bytes()];

    // ---- Q: Q_SUB tiles x 4 slices of 32 contiguous fp8 bytes -----------
    // Q arrives as BF16 [N, H, 448] and is packed in registers.
    // One exponent per head, not per 32.  The mfma's E8M0 scale partition is
    // not lane-local -- a lane's byte governs 16 of its own elements plus 16 of
    // a neighbour's -- so an in-register per-32 pack would have to reproduce
    // that partition exactly; a uniform byte per head makes it irrelevant.
    // The loaded values stay bf16 vectors on purpose.  Routing them through an
    // i32x4 and indexing dwords out of it silently delivers only one dword in
    // four (d = 0,1 mod 8 correct, the rest garbage).
    typedef unsigned short u16x8  __attribute__((ext_vector_type(8)));
    typedef short          s16x2  __attribute__((ext_vector_type(2)));
    typedef __bf16         bf16x2 __attribute__((ext_vector_type(2)));

    auto g_qb = make_gmem(reinterpret_cast<const bf16_t*>(kargs.q_ptr)
                              + (size_t)q_token * kargs.stride_q_n
                              + (size_t)h_base * kargs.stride_q_h,
                          (unsigned)((kargs.H - h_base) * kargs.stride_q_h * 2));

    i32x8 v_q[T::Q_SUB][T::D_SLICES];
    int   q_sb[T::Q_SUB];
#pragma unroll
    for (int qs = 0; qs < T::Q_SUB; ++qs) {
        // element j of lane (c,g) carries d = 128*ds + 64*(j/16) + 16*g + (j%16),
        // so each slice is two 16-element chunks 64 apart
        const int q_row = (h_wave + qs * T::Q_TILE + c) * kargs.stride_q_h;
        const int q_off = q_row + 16 * g;

        bf16x8 raw[T::D_SLICES][4];
#pragma unroll
        for (int ds = 0; ds < T::D_SLICES; ++ds)
#pragma unroll
            for (int k = 0; k < 4; ++k)
                raw[ds][k] = __builtin_bit_cast(bf16x8,
                    load<8>(g_qb, q_off + ds * 128 + (k >> 1) * 64 + (k & 1) * 8, 0, number<PA_NT_AUX>{}));

        // absmax in the integer domain: bf16 with the sign masked orders exactly
        // like magnitude, so no widening is needed.  k >= 2 of the last slice is
        // d = 448..511 -- past the real NoPE width -- and is excluded.
        u16x8 acc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int ds = 0; ds < T::D_SLICES; ++ds)
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                if (ds == T::D_SLICES - 1 && k >= 2) continue;
                acc = __builtin_elementwise_max(
                    acc, __builtin_bit_cast(u16x8, raw[ds][k]) & (unsigned short)0x7FFF);
            }
        unsigned short mxs = 0;
#pragma unroll
        for (int w = 0; w < 8; ++w) mxs = acc[w] > mxs ? acc[w] : mxs;
        // a head's 448 dims live in lanes {c, c+16, c+32, c+48}: two xor steps
        int mx = (int)mxs;
        int par = __builtin_amdgcn_ds_bpermute(((lane ^ 16) & 63) << 2, mx);
        mx = mx > par ? mx : par;
        par = __builtin_amdgcn_ds_bpermute(((lane ^ 32) & 63) << 2, mx);
        mx = mx > par ? mx : par;
        const float amax = __builtin_bit_cast(float, ((unsigned)mx) << 16);

        // ceil(log2(amax/240)) + 127 straight off the exponent field -- no
        // log2f/ceilf, and no 1-ULP error to push amax*inv past e4m3's 448.
        // 240 rather than 448 leaves the headroom the PV's P-gain assumes.
        int e = 127;
        if (amax > 0.f) {
            const unsigned bits = __builtin_bit_cast(unsigned, amax / 240.0f);
            e = (int)((bits >> 23) & 0xFFu) + (int)((bits & 0x7FFFFFu) != 0u);
            e = e < 1 ? 1 : (e > 200 ? 200 : e);
        }
        // the decode scale 2^(e-127): v_cvt_scalef32_* takes the factor that
        // will be applied on the way back out and divides by it going in.
        const float dec = __builtin_bit_cast(float, (unsigned)e << 23);
        q_sb[qs] = e | (e << 8) | (e << 16) | (e << 24);

#pragma unroll
        for (int ds = 0; ds < T::D_SLICES; ++ds) {
            i32x8 o;
#pragma unroll
            for (int k = 0; k < 4; ++k)
#pragma unroll
                for (int w = 0; w < 2; ++w) {
                    const bf16x2 a2 = {raw[ds][k][4 * w + 0], raw[ds][k][4 * w + 1]};
                    const bf16x2 b2 = {raw[ds][k][4 * w + 2], raw[ds][k][4 * w + 3]};
                    s16x2 p = {0, 0};
                    p = __builtin_amdgcn_cvt_scalef32_pk_fp8_bf16(p, a2, dec, false);
                    p = __builtin_amdgcn_cvt_scalef32_pk_fp8_bf16(p, b2, dec, true);
                    o[k * 2 + w] = __builtin_bit_cast(int, p);
                }
            v_q[qs][ds] = o;
        }
        // the upper half of slice 3 is d = 448..511, past the NoPE width.
        // Zeroing it is what lets the last slice run at full K=128 and makes the
        // K-side tail harmless.
#pragma unroll
        for (int t = 4; t < 8; ++t) v_q[qs][T::D_SLICES - 1][t] = 0;
    }

    // Q RoPE: 16 bytes per k-step at d = 32*st + 8*g
    auto g_qr = make_gmem(reinterpret_cast<const bf16_t*>(kargs.q_rope_ptr)
                              + (size_t)q_token * kargs.H * kargs.stride_qr_h
                              + (size_t)h_base * kargs.stride_qr_h,
                          (unsigned)((kargs.H - h_base) * kargs.stride_qr_h * 2));
    bf16x8 v_qr[T::Q_SUB][T::ROPE_KST];
#pragma unroll
    for (int qs = 0; qs < T::Q_SUB; ++qs)
#pragma unroll
        for (int st = 0; st < T::ROPE_KST; ++st)
            v_qr[qs][st] = __builtin_bit_cast(bf16x8,
                load<8>(g_qr, (h_wave + qs * T::Q_TILE + c) * kargs.stride_qr_h
                                  + 32 * st + 8 * g, 0, number<PA_NT_AUX>{}));

    const float c_row = kargs.softmax_scale * LOG2_E;

    f32x4 v_o[T::Q_SUB][T::O_LEN];
#if PA_ACC_LDS_INIT
    static_assert((T::Q_SUB * T::O_LEN) % PA_ACC_INIT_BATCH == 0,
                  "accumulator init batch must divide the accumulator count");
    static_assert(PA_ACC_INIT_KEEP >= 0
                      && PA_ACC_INIT_KEEP + PA_ACC_INIT_REFILL <= 63,
                  "invalid rolling LDS initialization window");
    const u32_t acc_init_addr = static_cast<u32_t>(
        reinterpret_cast<__UINTPTR_TYPE__>(smem + tid * sizeof(f32x4)));
    ds_write_vgpr(acc_init_addr, f32x4{0.f, 0.f, 0.f, 0.f});
    s_waitcnt_lgkmcnt(0_I);
#pragma unroll
    for (int qs = 0; qs < T::Q_SUB; ++qs)
#pragma unroll
        for (int j = 0; j < T::O_LEN; ++j) {
            v_o[qs][j] = ds_read_acc(acc_init_addr);
            constexpr int total = T::Q_SUB * T::O_LEN;
            const int issued = qs * T::O_LEN + j + 1;
            if (issued == total)
                s_waitcnt_lgkmcnt(0_I);
            else if constexpr (PA_ACC_INIT_KEEP == 0) {
                if ((issued % PA_ACC_INIT_BATCH) == 0)
                    s_waitcnt_lgkmcnt(0_I);
            } else {
                if (issued >= PA_ACC_INIT_KEEP + PA_ACC_INIT_REFILL
                    && ((issued - PA_ACC_INIT_KEEP) % PA_ACC_INIT_REFILL) == 0)
                    s_waitcnt_lgkmcnt(number<PA_ACC_INIT_KEEP>{});
            }
        }
#else
#pragma unroll
    for (int qs = 0; qs < T::Q_SUB; ++qs)
#pragma unroll
        for (int j = 0; j < T::O_LEN; ++j) {
            v_o[qs][j] = f32x4{0.f, 0.f, 0.f, 0.f};
            pin_acc_vgpr(v_o[qs][j]);
        }
#endif
    float m_row[T::Q_SUB], l_row[T::Q_SUB], m_ref[T::Q_SUB];
#pragma unroll
    for (int qs = 0; qs < T::Q_SUB; ++qs) {
        // not -inf: the aiter build uses -ffast-math
        m_row[qs] = opus::numeric_limits<float>::lowest();
        m_ref[qs] = opus::numeric_limits<float>::lowest();
        l_row[qs] = 0.f;
    }

    auto g_sink = make_gmem(kargs.attn_sink_ptr, (unsigned)(kargs.H * 4));
    float sink_log2[T::Q_SUB];
#pragma unroll
    for (int qs = 0; qs < T::Q_SUB; ++qs)
        sink_log2[qs] = load(g_sink, h_base + h_wave + qs * T::Q_TILE + c)[0] * LOG2_E;

    {
        int b, len;
        if (kargs.kv_lens_prefix) {
            b   = q_token * kargs.kv_stride_q_prefix;
            len = kargs.kv_lens_prefix[q_token];
        } else {
            b   = kargs.kv_indptr_prefix[q_token];
            len = kargs.kv_indptr_prefix[q_token + 1] - b;
        }
        accumulate_segment(kargs, kargs.unified_kv_ptr, kargs.unified_kv_rope_ptr,
                           kargs.total_pages, kargs.kv_indices_prefix,
                           kargs.sgl_page_shift[0], kargs.sgl_rows_per_page[0],
                           kargs.sgl_scale_off[0], b, len,
                           smem, v_q, v_qr, q_sb, v_o, c_row, m_row, l_row, m_ref,
                           sink_log2);
    }
    {
        int b, len;
        if (kargs.kv_lens_extend) {
            b   = q_token * kargs.kv_stride_q_extend;
            len = kargs.kv_lens_extend[q_token];
        } else {
            b   = kargs.kv_indptr_extend[q_token];
            len = kargs.kv_indptr_extend[q_token + 1] - b;
        }
        accumulate_segment(kargs, kargs.kv_ptr, kargs.kv_rope_ptr,
                           kargs.total_tokens, kargs.kv_indices_extend,
                           kargs.sgl_page_shift[1], kargs.sgl_rows_per_page[1],
                           kargs.sgl_scale_off[1], b, len,
                           smem, v_q, v_qr, q_sb, v_o, c_row, m_row, l_row, m_ref,
                           sink_log2);
    }

    // ---- normalise and store ---------------------------------------------
    // l_row already carries the sink term and both it and v_o live in the m_ref
    // frame, so the whole epilogue is one reciprocal.
    int out_lane = (int)threadIdx.x % T::WARP_SIZE;
    asm volatile("" : "+v"(out_lane));  // do not extend lane/c/g across both segments
    const int out_c = out_lane & 15;
    const int out_g = out_lane >> 4;
    auto g_o = make_gmem(reinterpret_cast<bf16_t*>(kargs.out_ptr)
                             + (size_t)q_token * kargs.stride_o_n
                             + (size_t)h_base * kargs.stride_o_h,
                         (unsigned)((kargs.H - h_base) * kargs.stride_o_h * 2));

#if PA_NO_COLLAPSE
    const float inv_alpha = __builtin_exp2f(-7.f);   // MAXE = 127
#else
    const float inv_alpha = __builtin_exp2f((float)(*kargs.max_e_ptr - 134));
#endif

#if PA_EPI_LDS
    // Coalesced epilogue through the LDS transpose.
    {
        constexpr int EP_PITCH = PA_EPI_PITCH;    // bf16 per row: 512 + pad
        constexpr int EP_ROWS  = T::Q_TILE;       // 16 heads per sub-tile
        // No barrier: accumulate_segment already ends with one, precisely so
        // the next user of LDS cannot race a wave still in its last PV, and
        // that covers this staging area too.  A zero-length segment returns
        // before that barrier, but then it has not written LDS either and the
        // previous segment's barrier still stands.  This area aliases the KV
        // tile, so if that trailing barrier is ever removed the read-back comes
        // out as ~8e34 garbage rather than as a small error.
        static_assert(T::smem_size_bytes()
                          >= (size_t)T::NUM_WARPS * EP_ROWS * EP_PITCH * 2,
                      "epilogue staging overruns LDS");
        auto s_ep = make_smem(reinterpret_cast<bf16_t*>(smem)
                              + warp * (EP_ROWS * EP_PITCH));
#pragma unroll
        for (int qs = 0; qs < T::Q_SUB; ++qs) {
            const float o_scale = (l_row[qs] > 0.f) ? (1.f / l_row[qs]) : 0.f;
#if PA_ACC_LDS_EPI
            // AGPRs cannot feed VALU, but they can feed LDS stores directly.
            // Transpose the whole f32 C layout through LDS, then normalise in
            // the coalesced readback.  This is both the AGPR->VGPR bridge and
            // the output transpose; a separate bf16 staging pass would add
            // another 48 LDS instructions per query sub-tile.
            constexpr int ACC_PITCH = PA_ACC_EPI_PITCH;  // f32 per head row
            static_assert(T::NUM_WARPS * EP_ROWS * ACC_PITCH * (int)sizeof(float)
                              <= (int)T::smem_size_bytes(),
                          "f32 accumulator transpose overruns LDS");
            const u32_t acc_write_addr = static_cast<u32_t>(
                reinterpret_cast<__UINTPTR_TYPE__>(
                    smem + ((warp * EP_ROWS + out_c) * ACC_PITCH
                            + out_g * 4) * sizeof(float)));
            static_for<T::O_LEN>([&](auto i_ot) {
                constexpr int ot = i_ot.value;
                ds_write_acc<ot * 16 * sizeof(float)>(
                    acc_write_addr, v_o[qs][ot]);
            });
            s_waitcnt_lgkmcnt(0_I);

            const u32_t acc_read_base = static_cast<u32_t>(
                reinterpret_cast<__UINTPTR_TYPE__>(
                    smem + (warp * EP_ROWS * ACC_PITCH
                            + out_lane * 8) * sizeof(float)));
#if PA_ACC_EPI_PIPE
            f32x4 staged_lo[2], staged_hi[2];
            unsigned staged_scale[2];
            auto issue_row = [&]<int HH, int SLOT>() {
                staged_scale[SLOT] = __builtin_amdgcn_ds_bpermute(
                    HH * 4, __builtin_bit_cast(unsigned, o_scale));
                staged_lo[SLOT] =
                    ds_read_vgpr<HH * ACC_PITCH * sizeof(float)>(acc_read_base);
                staged_hi[SLOT] = ds_read_vgpr<
                    HH * ACC_PITCH * sizeof(float) + sizeof(f32x4)>(acc_read_base);
            };

            issue_row.template operator()<0, 0>();
            static_for<EP_ROWS>([&](auto i_hh) {
                constexpr int hh = i_hh.value;
                constexpr int slot = hh & 1;
                s_waitcnt_lgkmcnt(0_I);
                asm volatile("" : "+v"(staged_lo[slot]), "+v"(staged_hi[slot]),
                                  "+v"(staged_scale[slot]) ::);

                vector_t<float, 8> value;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    value[i] = staged_lo[slot][i];
                    value[i + 4] = staged_hi[slot][i];
                }
                const float row_scale =
                    __builtin_bit_cast(float, staged_scale[slot]);
                const float os = (out_lane < T::D_NOPE / 8)
                                   ? row_scale * inv_alpha
                                   : row_scale;
                vector_t<float, 8> scaled = value * os;
                asm volatile("" : "+v"(scaled) :: "memory");

                if constexpr (hh + 1 < EP_ROWS)
                    issue_row.template operator()<hh + 1, slot ^ 1>();
                asm volatile("" ::: "memory");
                store<8>(g_o, cast<bf16_t>(scaled),
                         (h_wave + qs * T::Q_TILE + hh) * kargs.stride_o_h
                             + out_lane * 8, 0, number<PA_NT_AUXO>{});
            });
#else
            static_for<EP_ROWS>([&](auto i_hh) {
                constexpr int hh = i_hh.value;
                unsigned row_scale_bits = __builtin_amdgcn_ds_bpermute(
                    hh * 4, __builtin_bit_cast(unsigned, o_scale));
                f32x4 lo =
                    ds_read_vgpr<hh * ACC_PITCH * sizeof(float)>(acc_read_base);
                f32x4 hi = ds_read_vgpr<
                    hh * ACC_PITCH * sizeof(float) + sizeof(f32x4)>(acc_read_base);
                s_waitcnt_lgkmcnt(0_I);
                asm volatile("" : "+v"(lo), "+v"(hi), "+v"(row_scale_bits) ::);

                vector_t<float, 8> value;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    value[i] = lo[i];
                    value[i + 4] = hi[i];
                }
                const float row_scale = __builtin_bit_cast(float, row_scale_bits);
                const float os = (out_lane < T::D_NOPE / 8)
                                   ? row_scale * inv_alpha
                                   : row_scale;
                store<8>(g_o, cast<bf16_t>(value * os),
                         (h_wave + qs * T::Q_TILE + hh) * kargs.stride_o_h
                             + out_lane * 8, 0, number<PA_NT_AUXO>{});
            });
#endif
#else
            static_for<T::D_TILE / 16>([&](auto i_ot) {
                constexpr int ot = i_ot.value;
                pin_acc_vgpr(v_o[qs][ot]);
                // the RoPE columns never went through the P gain, so they must
                // not be divided by it
                const float os = (ot < T::O_TILES) ? (o_scale * inv_alpha)
                                                   : o_scale;
                s_ep.template store<4>(
                    cast<bf16_t>(__builtin_bit_cast(vector_t<float, 4>,
                                                    v_o[qs][ot] * os)),
                    out_c * EP_PITCH + ot * 16 + out_g * 4);
            });
#endif
            // The 16 rows a wave reads back are the 16 it just wrote, so this
            // only has to order a wave against itself -- lgkmcnt, not a barrier.
#if !PA_ACC_LDS_EPI
            s_waitcnt_lgkmcnt(0_I);
#pragma unroll
            for (int hh = 0; hh < EP_ROWS; ++hh) {
                // 64 lanes x 8 bf16 = one whole 1024 B head row per store, so a
                // row takes one instruction and there is no offset to fit in
                // the 12-bit immediate at all.
                //
                // load<>, not _load<>: the underscore forms take byte offsets
                // and the plain ones take elements.  Mixing them is silent --
                // it compiles, and reads a row and a half past the end.
                store<8>(g_o, s_ep.template load<8>(hh * EP_PITCH + out_lane * 8),
                         (h_wave + qs * T::Q_TILE + hh) * kargs.stride_o_h
                             + out_lane * 8, 0, number<PA_NT_AUXO>{});
            }
            s_waitcnt_lgkmcnt(0_I);   // the rows are reused by the next qs
#endif
        }
    }
#else
#pragma unroll
    for (int qs = 0; qs < T::Q_SUB; ++qs) {
        const float o_scale = (l_row[qs] > 0.f) ? (1.f / l_row[qs]) : 0.f;
        const int o_off =
            (h_wave + qs * T::Q_TILE + out_c) * kargs.stride_o_h + 4 * out_g;

        static_for<T::D_TILE / 16>([&](auto i_ot) {   // 28 NoPE + 4 RoPE subtiles
            constexpr int ot = i_ot.value;
            pin_acc_vgpr(v_o[qs][ot]);
            // the RoPE columns never went through the P gain, so they must not
            // be divided by it
            const float os = (ot < T::O_TILES) ? (o_scale * inv_alpha) : o_scale;
            const f32x4 t4 = v_o[qs][ot] * os;
            store<4>(g_o, cast<bf16_t>(__builtin_bit_cast(vector_t<float, 4>, t4)),
                     o_off + ot * 16, 0, number<PA_NT_AUXO>{});
        });
    }
#endif

}

#endif // gfx950 device pass

}  // namespace pa_sparse_mla
