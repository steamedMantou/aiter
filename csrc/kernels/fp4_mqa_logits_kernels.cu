// SPDX-License-Identifier: MIT
// FP4 (MXFP4) paged MQA-logits HIP kernel for gfx950 (CDNA4).
//
// logits[b*N+n, j] = sum_h relu( Q[b,n,h,:] . K[tok(j),:] ) * w[b*N+n, h]
// with tok(j) = block_table[b, j], causal limit j <= ctx[b] - N + n.
//
// Uses the scaled FP4 matrix core v_mfma_scale_f32_16x16x128_f8f6f4
// (one instruction contracts the full D=128 with 4 e8m0 micro-scale blocks).
//
// Per warp / per 16-token tile:
//   * One MFMA per 16-head tile (H/16 tiles) gives s[head, token].
//   * relu * weight is reduced over heads: 4 in-register heads * H/16 tiles
//     per lane, then a 2-shuffle cross-lane sum over the 4 lane groups.
//   * 16 KV logits are produced per tile.
//
// KV is MXFP4 and fused per token as: [D/2 e2m1 bytes | D/32 e8m0 bytes | pad]
// row-padded to 16B so the 16-byte operand loads stay aligned.

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdlib>

typedef int          int4v   __attribute__((ext_vector_type(4)));
typedef int          int8v   __attribute__((ext_vector_type(8)));
typedef float        float2v __attribute__((ext_vector_type(2)));
typedef float        float4v __attribute__((ext_vector_type(4)));
typedef float        float16v __attribute__((ext_vector_type(16)));
typedef unsigned int u32x2   __attribute__((ext_vector_type(2)));

#ifndef WARPS
#define WARPS 4
#endif
#ifndef UNROLL
#define UNROLL 2        // tiles processed per warp iteration (2*UNROLL gathers in flight)
#endif
#ifndef TPW
#define TPW 1           // ldsk: tiles each warp gathers per chunk (ILP between barriers)
#endif
#define LANES 64
#define NEG_INF (-__builtin_huge_valf())

__device__ __forceinline__ int8v load16(const uint8_t* p) {
    int8v v;
    const int4 q = *reinterpret_cast<const int4*>(p);  // 16 aligned bytes
    v[0] = q.x; v[1] = q.y; v[2] = q.z; v[3] = q.w;
    v[4] = 0; v[5] = 0; v[6] = 0; v[7] = 0;
    return v;
}

// Store only the 16 FP4 bytes a lane actually owns (4 i32). The MFMA operand is
// v8i32, but for the f4 format only the low 16 bytes are read, so we widen with
// zero padding right at the instruction and keep persistent storage at 4 VGPR.
__device__ __forceinline__ int4v load16v4(const uint8_t* p) {
    const int4 q = *reinterpret_cast<const int4*>(p);
    return int4v{q.x, q.y, q.z, q.w};
}
__device__ __forceinline__ int8v widen(int4v a) {
    return int8v{a[0], a[1], a[2], a[3], 0, 0, 0, 0};
}

// Sum a value across the 4 lane-groups {l, l^16, l^32, l^48} that share a token.
//
// Default uses __shfl_xor, which lowers to ds_bpermute. That is the FASTEST
// choice here: this kernel allocates no LDS, so the ds (LDS-crossbar) unit is
// otherwise idle and the permute overlaps for free with the VMEM-bound tile
// loop. Measured A/B (tests/ab_swaplane.py) shows the CDNA4 cross-lane swap ALU
// path (-DFP4_USE_PERMLANE, below) is equal at large batch but ~0.75-0.95x at
// small batch, because it adds VALU pressure (2 permlane + 2 cndmask) to a loop
// that already has the relu*weight VALU work, while leaving the ds unit idle.
// The permlane path is kept (verified lane-equivalent in probe_permlane.hip) for
// LDS-heavy / compute-bound variants where the trade-off flips.
__device__ __forceinline__ float group_sum4(float part, int lane) {
#if defined(__gfx950__) && defined(FP4_USE_PERMLANE)
    unsigned int b = __builtin_bit_cast(unsigned int, part);
    u32x2 r = __builtin_amdgcn_permlane16_swap(b, b, false, false);
    part += __builtin_bit_cast(float, (lane & 16) ? r[0] : r[1]);  // ^16
    b = __builtin_bit_cast(unsigned int, part);
    r = __builtin_amdgcn_permlane32_swap(b, b, false, false);
    part += __builtin_bit_cast(float, (lane & 32) ? r[0] : r[1]);  // ^32
    return part;
#else
    part += __shfl_xor(part, 16);
    part += __shfl_xor(part, 32);
    return part;
#endif
}

// VALU (cross-lane permute) variant of the reduction. The ldsk kernel keeps the
// ds (LDS) unit busy staging K, so doing the 4-group sum on the ALU via the
// CDNA4 permlane swaps (verified lane-equivalent in tests/probe_permlane.hip)
// avoids contending for the ds crossbar. For the LDS-free kernels the opposite
// holds, so they keep the ds_bpermute path above.
__device__ __forceinline__ float group_sum4_valu(float part, int lane) {
#if defined(__gfx950__)
    unsigned int b = __builtin_bit_cast(unsigned int, part);
    u32x2 r = __builtin_amdgcn_permlane16_swap(b, b, false, false);
    part += __builtin_bit_cast(float, (lane & 16) ? r[0] : r[1]);  // ^16
    b = __builtin_bit_cast(unsigned int, part);
    r = __builtin_amdgcn_permlane32_swap(b, b, false, false);
    part += __builtin_bit_cast(float, (lane & 32) ? r[0] : r[1]);  // ^32
    return part;
#else
    part += __shfl_xor(part, 16);
    part += __shfl_xor(part, 32);
    return part;
#endif
}

// Packed in-lane relu*weight accumulate for one MFMA output (float4 acc = 4
// heads). Uses v_pk_max_f32 (single packed relu, no fmaxf NaN-canonicalize --
// logits are finite) + v_pk_fma_f32 into a 2-wide accumulator, halving the
// VALU vs the scalar 4x(v_max,v_max,v_fmac) path and breaking the long scalar
// part+= dependency chain. Caller sums the 2 lanes once at the end.
// Fully-packed relu*weight accumulate. gfx950 has NO v_pk_max_f32, but it has
// packed add/mul/fma, so we use the identity relu(x) = 0.5*(x + |x|): |x| is a
// free source modifier, x+|x| is one v_pk_add_f32, and the 0.5 is folded out to
// a single final scale (caller). Net: the relu becomes ONE packed add (vs two
// scalar v_max), and the weighted accumulate is one v_pk_fma into a 2-wide
// accumulator -- so the per-pair VALU drops from {2x v_max + fma(s)} to
// {v_pk_add + v_pk_fma}, and the long scalar part+= chain is broken.
// Caller must apply the 0.5: part = 0.5f * (pacc[0] + pacc[1]).
__device__ __forceinline__ void fp4_relu_w_pk(float4v acc, const float* w,
                                              float2v& pacc) {
    float2v x_lo = {acc[0], acc[1]};
    float2v x_hi = {acc[2], acc[3]};
    float2v t_lo = x_lo + __builtin_elementwise_abs(x_lo);   // v_pk_add + abs mod
    float2v t_hi = x_hi + __builtin_elementwise_abs(x_hi);
    pacc += t_lo * (float2v){w[0], w[1]};                    // v_pk_fma
    pacc += t_hi * (float2v){w[2], w[3]};
}

// Wave priority hint (gluon-style): raise priority around the MFMA-heavy region
// so the scheduler crunches the matrix core ahead of the cross-lane reduction /
// store epilogue, then drop it. Pure hint (no scheduling fence), so it never
// changes results and is safe to leave compiled-in behind a template flag.
template <int P>
__device__ __forceinline__ void fp4_setprio() {
#if defined(__gfx950__)
    __builtin_amdgcn_s_setprio(P);   // builtin requires a compile-time constant
#endif
}

// HEADS is a template param so the per-lane head loop unrolls and the
// Q/weight register arrays are sized at compile time.
//
// LAYOUT selects the KV-cache layout at compile time (zero runtime branch), to
// match the two layouts the FP8 "gluon" deepgemm_fp8_paged_mqa_logits accepts:
//
//   LAYOUT_INTERLEAVED (non-preshuffle, KVBlockSize==1): the FP8-gluon
//     non-preshuffle layout. One physical block == one token; per-token
//     block_tables. Each row is the fused MXFP4 token [D/2 e2m1 | D/32 e8m0 | pad]
//     (kv_stride bytes). krow = kv + block_tables[b,jj]*kv_stride.
//
//   LAYOUT_SEGREGATED (non-preshuffle, KVBlockSize>1): segregated paged cache,
//     block_tables maps logical->physical block (granularity = block_size
//     tokens). Within a block: data region [0, block_size*DP) token-major, then
//     scale region [block_size*DP, block_size*(DP+DS)) token-major.
//     blk_stride = bytes per physical block (= kv_cache.stride(0), padding incl).
//
//   LAYOUT_PRESHUFFLE (KVBlockSize multiple of 16): the FP8-gluon preshuffle
//     layout. Same segregated data|scale regions and per-block block_tables, but
//     the data region is swizzled by shuffle_weight(layout=(16,16)) over the
//     [block_size, DP] byte matrix so each lane reads 16 contiguous bytes:
//       byte(pos, lg) = (pos/16)*16*DP + lg*256 + (pos%16)*16 .
//     The scale region stays token-major (un-shuffled), mirroring the FP8 kernel.
enum { LAYOUT_INTERLEAVED = 0, LAYOUT_SEGREGATED = 1, LAYOUT_PRESHUFFLE = 2 };

template <int HEADS, int LAYOUT = LAYOUT_INTERLEAVED,
          int UNR = UNROLL, int NW = WARPS, bool SCHED = false>
__global__ void fp4_pa_mqa_kernel(
    const uint8_t* __restrict__ q_p,    // [B,N,H,D/2]
    const uint8_t* __restrict__ q_s,    // [B,N,H,D/32]
    const uint8_t* __restrict__ kv,     // interleaved: [num_blocks, kv_stride]
                                        // seg/preshuffle: [num_blocks, blk_stride]
    const int*     __restrict__ block_tables,  // [B, max_block_len]
    const float*   __restrict__ weights,       // [B*N, H]
    float*         __restrict__ out,           // [B*N, max_model_len]
    const int*     __restrict__ context_lens,  // [B]
    int B, int N, int max_model_len, int max_block_len,
    int kv_stride, int split_kv,
    int block_size, int blk_stride)     // seg/preshuffle only (else ignored)
{
    constexpr int DP = 64;          // D/2 packed bytes  (D = 128)
    constexpr int DS = 4;           // D/32 scale blocks
    constexpr int HT = HEADS / 16;  // head tiles

    const int pid = blockIdx.x;
    const int nn  = pid % N;
    int rem       = pid / N;
    const int b   = rem % B;
    const int split = rem / B;

    const int ctx = context_lens[b];
    if (ctx <= 0) return;

    const int total_tiles = (ctx + 15) / 16;
    const int tiles_per_split = (total_tiles + split_kv - 1) / split_kv;
    const int tile_start = split * tiles_per_split;
    const int tile_end   = min(tile_start + tiles_per_split, total_tiles);
    if (tile_start >= tile_end) return;

    const int warp = threadIdx.x / LANES;
    const int lane = threadIdx.x % LANES;
    const int lt   = lane & 15;     // token / input-row index
    const int lg   = lane >> 4;     // kgroup (input) / head-group (output)

    // ---- Load Q + scales + weights once (reused over all token tiles) ----
    const int qn = (b * N + nn);
    const uint8_t* q_p_base = q_p + (size_t)qn * HEADS * DP;
    const uint8_t* q_s_base = q_s + (size_t)qn * HEADS * DS;
    const float*   w_base   = weights + (size_t)qn * HEADS;

    int4v qreg[HT];
    int   qscale[HT];
    float wreg[HT][4];
#pragma unroll
    for (int ht = 0; ht < HT; ++ht) {
        const int head_in = 16 * ht + lt;                 // input row = lt
        qreg[ht]   = load16v4(q_p_base + (size_t)head_in * DP + 16 * lg);
        qscale[ht] = (int)q_s_base[(size_t)head_in * DS + lg];
#pragma unroll
        for (int r = 0; r < 4; ++r)
            wreg[ht][r] = w_base[16 * ht + 4 * lg + r];   // output head = 16ht+4lg+r
    }

    const int limit   = ctx - N + nn;
    const size_t bt_row = (size_t)b * max_block_len;
    float* out_row = out + (size_t)qn * max_model_len;

    auto gather = [&](int t_idx, int4v& bK, int& sbK) {
        const int jj = t_idx * 16 + lt;
        if constexpr (LAYOUT == LAYOUT_INTERLEAVED) {
            const int blk = (jj < ctx) ? block_tables[bt_row + jj] : 0;
            const uint8_t* krow = kv + (size_t)blk * kv_stride;
            bK = load16v4(krow + 16 * lg);
            sbK = (int)krow[DP + lg];
        } else {
            // segregated / preshuffle: logical->physical block, then in-block offset.
            const int phys = (jj < ctx) ? block_tables[bt_row + jj / block_size] : 0;
            const uint8_t* base = kv + (size_t)phys * blk_stride;
            const int pos = jj % block_size;
            if constexpr (LAYOUT == LAYOUT_PRESHUFFLE) {
                // shuffle_weight(16,16) over [block_size, DP]: the 16 bytes a lane
                // owns (db = 16*lg + 0..15) are contiguous at this offset.
                bK = load16v4(base + (size_t)(pos / 16) * (16 * DP)
                                   + (size_t)lg * 256 + (size_t)(pos % 16) * 16);
            } else {
                bK = load16v4(base + (size_t)pos * DP + 16 * lg);  // data region
            }
            sbK = (int)base[(size_t)block_size * DP + pos * DS + lg];  // scale region
        }
    };

    auto compute = [&](int t_idx, int4v bK, int sbK) {
#if defined(FP4_MEMONLY)
        // probe: identical gather, no MFMA -> isolates pure paged-gather BW
        float part = (float)(bK[0] ^ bK[1] ^ bK[2] ^ bK[3] ^ sbK);
#else
        const int8v Kop = widen(bK);
        float part = 0.0f;
#pragma unroll
        for (int ht = 0; ht < HT; ++ht) {
            float4v acc = {0.f, 0.f, 0.f, 0.f};
            acc = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
                widen(qreg[ht]), Kop, acc, 4, 4, 0, qscale[ht], 0, sbK);
#pragma unroll
            for (int r = 0; r < 4; ++r)
                part += fmaxf(acc[r], 0.0f) * wreg[ht][r];
        }
#endif
        part = group_sum4(part, lane);   // reduce over the 4 lane groups
        const int j = t_idx * 16 + lt;
        if (lane < 16 && j < max_model_len)
            out_row[j] = (j <= limit) ? part : NEG_INF;
    };

    // UNR-wide, 2-deep software pipeline: keep 2*UNR KV gathers in flight per
    // warp so HBM stays busy (this kernel is memory-latency bound). NW (warps
    // per CTA) and UNR (pipeline depth) are template knobs so the segregated /
    // interleaved paths can be tuned per shape exactly like the preshuffle path.
    const int it0 = tile_start + warp;
    int4v Kc[UNR]; int Sc[UNR];
#pragma unroll
    for (int u = 0; u < UNR; ++u) {
        const int t = it0 + u * NW;
        if (t < tile_end) gather(t, Kc[u], Sc[u]);
    }
    for (int it = it0; it < tile_end; it += UNR * NW) {
        int4v Kn[UNR]; int Sn[UNR];
#pragma unroll
        for (int u = 0; u < UNR; ++u) {
            const int t = it + (UNR + u) * NW;
            if (t < tile_end) gather(t, Kn[u], Sn[u]);
        }
        if constexpr (SCHED) fp4_setprio<1>();   // prioritize MFMA over epilogue
#pragma unroll
        for (int u = 0; u < UNR; ++u) {
            const int t = it + u * NW;
            if (t < tile_end) compute(t, Kc[u], Sc[u]);
        }
        if constexpr (SCHED) fp4_setprio<0>();
#pragma unroll
        for (int u = 0; u < UNR; ++u) { Kc[u] = Kn[u]; Sc[u] = Sn[u]; }
    }
}

// ---- dedicated preshuffle kernel -----------------------------------------
// Specialized FP4 path for the FP8-gluon preshuffle KV layout. KVBLK (tokens
// per physical block, a multiple of 16) is a compile-time constant, which lets
// us drop the per-lane integer div/mod and the redundant per-lane block_tables
// gather that the generic LAYOUT_PRESHUFFLE branch pays:
//   * one 16-token tile lives entirely in ONE physical block, so the physical
//     block id depends only on the tile index -> a single SCALAR load per tile
//     (broadcast to all lanes) instead of 16 identical per-lane loads;
//   * the in-block 16-group rank (rank16) and the swizzled data offset are pure
//     compile-time-strided expressions (no `% block_size` / `/ block_size`);
//   * tiles are bounded by total_tiles == ceil(ctx/16) and the per-block table
//     has ceil(maxctx/KVBLK) entries, so the per-lane `jj < ctx` OOB guard in
//     the gather is unnecessary (masking still happens at store time).
// In the latency-bound decode regime these removed scalar/VALU ops are what
// closed the gap to (and past) the FP8 gluon preshuffle kernel.
template <int HEADS, int KVBLK, int UNR = UNROLL, int NW = WARPS, bool SCHED = false>
__global__ __launch_bounds__(NW * LANES) void fp4_pa_mqa_preshuffle_kernel(
    const uint8_t* __restrict__ q_p,    // [B,N,H,D/2]
    const uint8_t* __restrict__ q_s,    // [B,N,H,D/32]
    const uint8_t* __restrict__ kv,     // [num_blocks, blk_stride] preshuffled
    const int*     __restrict__ block_tables,  // [B, max_block_len] per-block
    const float*   __restrict__ weights,       // [B*N, H]
    float*         __restrict__ out,           // [B*N, max_model_len]
    const int*     __restrict__ context_lens,  // [B]
    int B, int N, int max_model_len, int max_block_len,
    int kv_stride, int split_kv,
    int block_size, int blk_stride)     // block_size == KVBLK (runtime mirror)
{
    constexpr int DP = 64;          // D/2 packed bytes  (D = 128)
    constexpr int DS = 4;           // D/32 scale blocks
    constexpr int HT = HEADS / 16;  // head tiles
    constexpr int K16 = KVBLK / 16; // 16-token groups per physical block

    const int pid = blockIdx.x;
    const int nn  = pid % N;
    int rem       = pid / N;
    const int b   = rem % B;
    const int split = rem / B;

    const int ctx = context_lens[b];
    if (ctx <= 0) return;

    const int total_tiles = (ctx + 15) / 16;
    const int tiles_per_split = (total_tiles + split_kv - 1) / split_kv;
    const int tile_start = split * tiles_per_split;
    const int tile_end   = min(tile_start + tiles_per_split, total_tiles);
    if (tile_start >= tile_end) return;

    const int warp = threadIdx.x / LANES;
    const int lane = threadIdx.x % LANES;
    const int lt   = lane & 15;     // token within tile / Q input-row
    const int lg   = lane >> 4;     // kgroup (input) / head-group (output)

    // ---- Load Q + scales + weights once (reused over all token tiles) ----
    const int qn = (b * N + nn);
    const uint8_t* q_p_base = q_p + (size_t)qn * HEADS * DP;
    const uint8_t* q_s_base = q_s + (size_t)qn * HEADS * DS;
    const float*   w_base   = weights + (size_t)qn * HEADS;

    int4v qreg[HT];
    int   qscale[HT];
    float wreg[HT][4];
#pragma unroll
    for (int ht = 0; ht < HT; ++ht) {
        const int head_in = 16 * ht + lt;
        qreg[ht]   = load16v4(q_p_base + (size_t)head_in * DP + 16 * lg);
        qscale[ht] = (int)q_s_base[(size_t)head_in * DS + lg];
#pragma unroll
        for (int r = 0; r < 4; ++r)
            wreg[ht][r] = w_base[16 * ht + 4 * lg + r];
    }

    const int limit   = ctx - N + nn;
    const size_t bt_row = (size_t)b * max_block_len;
    float* out_row = out + (size_t)qn * max_model_len;

    // Per-lane in-block byte offsets are fixed across the whole tile loop (they
    // depend only on lt/lg), so hoist them out of the gather.
    const size_t data_off  = (size_t)lg * 256 + (size_t)lt * 16;  // + rank16*16*DP
    const int    scale_col = lt * DS + lg;                         // + rank16*16*DS

    auto gather = [&](int t_idx, int4v& bK, int& sbK) {
        const int logical_blk = t_idx / K16;              // K16 compile-time
        const int rank16      = t_idx - logical_blk * K16;  // t_idx % K16
        const int phys = block_tables[bt_row + logical_blk];  // scalar (uniform)
        const uint8_t* base = kv + (size_t)phys * blk_stride;
        bK  = load16v4(base + (size_t)rank16 * (16 * DP) + data_off);
        sbK = (int)base[(size_t)KVBLK * DP + (size_t)rank16 * (16 * DS) + scale_col];
    };

    auto compute = [&](int t_idx, int4v bK, int sbK) {
#if defined(FP4_MEMONLY)
        float part = (float)(bK[0] ^ bK[1] ^ bK[2] ^ bK[3] ^ sbK);  // isolate gather BW
#else
        const int8v Kop = widen(bK);
        float2v pacc = {0.0f, 0.0f};   // packed relu + weight FMA accumulator
#pragma unroll
        for (int ht = 0; ht < HT; ++ht) {
            float4v acc = {0.f, 0.f, 0.f, 0.f};
            acc = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
                widen(qreg[ht]), Kop, acc, 4, 4, 0, qscale[ht], 0, sbK);
            fp4_relu_w_pk(acc, wreg[ht], pacc);
        }
        float part = 0.5f * (pacc[0] + pacc[1]);   // fold the relu 0.5 once
#endif
        part = group_sum4(part, lane);
        const int j = t_idx * 16 + lt;
        if (lane < 16 && j < max_model_len)
            out_row[j] = (j <= limit) ? part : NEG_INF;
    };

    const int it0 = tile_start + warp;
    int4v Kc[UNR]; int Sc[UNR];
#pragma unroll
    for (int u = 0; u < UNR; ++u) {
        const int t = it0 + u * NW;
        if (t < tile_end) gather(t, Kc[u], Sc[u]);
    }
    for (int it = it0; it < tile_end; it += UNR * NW) {
        int4v Kn[UNR]; int Sn[UNR];
#pragma unroll
        for (int u = 0; u < UNR; ++u) {
            const int t = it + (UNR + u) * NW;
            if (t < tile_end) gather(t, Kn[u], Sn[u]);
        }
        if constexpr (SCHED) fp4_setprio<1>();   // prioritize MFMA over epilogue
#pragma unroll
        for (int u = 0; u < UNR; ++u) {
            const int t = it + u * NW;
            if (t < tile_end) compute(t, Kc[u], Sc[u]);
        }
        if constexpr (SCHED) fp4_setprio<0>();
#pragma unroll
        for (int u = 0; u < UNR; ++u) { Kc[u] = Kn[u]; Sc[u] = Sn[u]; }
    }
}

// gluon-style explicit instruction-group scheduling: interleave global KV loads
// with the matrix core so VMEM latency is hidden behind MFMA, instead of relying
// on the default scheduler. Only used by the *_sched kernels below (compile-time
// opt-in) so the proven default kernels are untouched.
__device__ __forceinline__ void fp4_iglp() {
#if defined(__gfx950__)
    __builtin_amdgcn_iglp_opt(0);   // 0 = interleave MFMA with mem ops
#endif
}

// gluon-style manual instruction-group scheduling. Mask bits (match the gluon
// kernel's iglp hints): VMEM/buffer_load=0x020, MFMA=0x008, VALU=0x002. A
// sequence of these emitted after issuing the loads+MFMAs of one pipeline stage
// forces the scheduler to interleave global KV prefetch with the matrix core
// (hiding VMEM latency) instead of clustering all loads then all MFMAs.
template <int MASK, int N>
__device__ __forceinline__ void fp4_sgb() {
#if defined(__gfx950__)
    __builtin_amdgcn_sched_group_barrier(MASK, N, 0);
#endif
}
#define FP4_SGB_VMEM(n) fp4_sgb<0x020, n>()
#define FP4_SGB_MFMA(n) fp4_sgb<0x008, n>()

// ---- dedicated segregated (non-preshuffle) kernel ------------------------
// Same compile-time-KVBLK specialization as the preshuffle kernel, for the
// non-swizzled segregated layout that sglang's fp4 indexer actually produces.
// The generic LAYOUT_SEGREGATED branch pays, per 16-token tile, 16 identical
// per-lane block_tables loads + a per-lane integer div/mod for jj%/jj/. Because
// KVBLK is a multiple of 16, a tile never straddles a physical block, so:
//   * the physical block id is uniform over the tile -> ONE scalar load;
//   * rank16 (tile's 16-group within the block) is a compile-time-strided
//     expression of t_idx (no runtime % / /);
//   * the per-lane in-block byte offset depends only on (lt,lg) -> hoisted.
// Data region is token-major: byte(pos,lg) = pos*DP + lg*16, pos = rank16*16+lt
// (this is the ONLY difference vs the preshuffle kernel, whose data is swizzled
// to lg*256 + (pos%16)*16). The scale region is token-major in both.
template <int HEADS, int KVBLK, int UNR = UNROLL, int NW = WARPS, bool SCHED = false>
__global__ __launch_bounds__(NW * LANES) void fp4_pa_mqa_seg_kernel(
    const uint8_t* __restrict__ q_p,    // [B,N,H,D/2]
    const uint8_t* __restrict__ q_s,    // [B,N,H,D/32]
    const uint8_t* __restrict__ kv,     // [num_blocks, blk_stride] segregated
    const int*     __restrict__ block_tables,  // [B, max_block_len] per-block
    const float*   __restrict__ weights,       // [B*N, H]
    float*         __restrict__ out,           // [B*N, max_model_len]
    const int*     __restrict__ context_lens,  // [B]
    int B, int N, int max_model_len, int max_block_len,
    int kv_stride, int split_kv,
    int block_size, int blk_stride)     // block_size == KVBLK (runtime mirror)
{
    constexpr int DP = 64;          // D/2 packed bytes  (D = 128)
    constexpr int DS = 4;           // D/32 scale blocks
    constexpr int HT = HEADS / 16;  // head tiles
    constexpr int K16 = KVBLK / 16; // 16-token groups per physical block

    const int pid = blockIdx.x;
    const int nn  = pid % N;
    int rem       = pid / N;
    const int b   = rem % B;
    const int split = rem / B;

    const int ctx = context_lens[b];
    if (ctx <= 0) return;

    const int total_tiles = (ctx + 15) / 16;
    const int tiles_per_split = (total_tiles + split_kv - 1) / split_kv;
    const int tile_start = split * tiles_per_split;
    const int tile_end   = min(tile_start + tiles_per_split, total_tiles);
    if (tile_start >= tile_end) return;

    const int warp = threadIdx.x / LANES;
    const int lane = threadIdx.x % LANES;
    const int lt   = lane & 15;
    const int lg   = lane >> 4;

    const int qn = (b * N + nn);
    const uint8_t* q_p_base = q_p + (size_t)qn * HEADS * DP;
    const uint8_t* q_s_base = q_s + (size_t)qn * HEADS * DS;
    const float*   w_base   = weights + (size_t)qn * HEADS;

    int4v qreg[HT];
    int   qscale[HT];
    float wreg[HT][4];
#pragma unroll
    for (int ht = 0; ht < HT; ++ht) {
        const int head_in = 16 * ht + lt;
        qreg[ht]   = load16v4(q_p_base + (size_t)head_in * DP + 16 * lg);
        qscale[ht] = (int)q_s_base[(size_t)head_in * DS + lg];
#pragma unroll
        for (int r = 0; r < 4; ++r)
            wreg[ht][r] = w_base[16 * ht + 4 * lg + r];
    }

    const int limit   = ctx - N + nn;
    const size_t bt_row = (size_t)b * max_block_len;
    float* out_row = out + (size_t)qn * max_model_len;

    // Token-major (segregated) per-lane offsets, fixed across the tile loop.
    const size_t data_off  = (size_t)lt * DP + (size_t)lg * 16;  // + rank16*16*DP
    const int    scale_col = lt * DS + lg;                        // + rank16*16*DS

    auto gather = [&](int t_idx, int4v& bK, int& sbK) {
        const int logical_blk = t_idx / K16;
        const int rank16      = t_idx - logical_blk * K16;
        const int phys = block_tables[bt_row + logical_blk];  // scalar (uniform)
        const uint8_t* base = kv + (size_t)phys * blk_stride;
        bK  = load16v4(base + (size_t)rank16 * (16 * DP) + data_off);
        sbK = (int)base[(size_t)KVBLK * DP + (size_t)rank16 * (16 * DS) + scale_col];
    };

    auto compute = [&](int t_idx, int4v bK, int sbK) {
#if defined(FP4_MEMONLY)
        float part = (float)(bK[0] ^ bK[1] ^ bK[2] ^ bK[3] ^ sbK);
#else
        const int8v Kop = widen(bK);
        float2v pacc = {0.0f, 0.0f};   // 2-wide packed accumulator
#pragma unroll
        for (int ht = 0; ht < HT; ++ht) {
            float4v acc = {0.f, 0.f, 0.f, 0.f};
            acc = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
                widen(qreg[ht]), Kop, acc, 4, 4, 0, qscale[ht], 0, sbK);
            fp4_relu_w_pk(acc, wreg[ht], pacc);   // packed relu + weight FMA
        }
        float part = 0.5f * (pacc[0] + pacc[1]);   // fold the relu 0.5 once
#endif
        part = group_sum4(part, lane);
        const int j = t_idx * 16 + lt;
        if (lane < 16 && j < max_model_len)
            out_row[j] = (j <= limit) ? part : NEG_INF;
    };

    const int it0 = tile_start + warp;
    int4v Kc[UNR]; int Sc[UNR];
#pragma unroll
    for (int u = 0; u < UNR; ++u) {
        const int t = it0 + u * NW;
        if (t < tile_end) gather(t, Kc[u], Sc[u]);
    }
    for (int it = it0; it < tile_end; it += UNR * NW) {
        int4v Kn[UNR]; int Sn[UNR];
#pragma unroll
        for (int u = 0; u < UNR; ++u) {
            const int t = it + (UNR + u) * NW;
            if (t < tile_end) gather(t, Kn[u], Sn[u]);
        }
        // s_set_prio over the MFMA region. NOTE measured immaterial here (1 vs 3
        // both timing-neutral): small batch has low occupancy (~2 waves/SIMD, no
        // arbitration to win) and large batch is HBM-bound (all waves memory-
        // stalled), so wave-priority cannot add bandwidth or cut the launch floor.
        if constexpr (SCHED) { fp4_setprio<1>(); fp4_iglp(); }
#pragma unroll
        for (int u = 0; u < UNR; ++u) {
            const int t = it + u * NW;
            if (t < tile_end) compute(t, Kc[u], Sc[u]);
        }
        if constexpr (SCHED) fp4_setprio<0>();
#pragma unroll
        for (int u = 0; u < UNR; ++u) { Kc[u] = Kn[u]; Sc[u] = Sn[u]; }
    }
}

// ---- gluon-asm-aligned segregated kernel ---------------------------------
// Mirrors the FP8 "gluon" preshuffle kernel's instruction schedule: an explicit
// software pipeline that prefetches the next tile's KV (buffer_load) while the
// matrix core works the current tile, with manual sched_group_barrier groups
// interleaving VMEM loads and MFMAs and an s_set_prio(3) priority wave around
// the math. Goal: match gluon's latency hiding so fp4 (fewer bytes / 1 MFMA per
// D=128) is >= as fast at every shape, including the tiny launch-floor ones.
// Same token-major segregated layout + MFMA/reduction lane layout as
// fp4_pa_mqa_seg_kernel; only the loop schedule differs.
template <int HEADS, int KVBLK, int UNR = 2, int NW = WARPS>
__global__ __launch_bounds__(NW * LANES) void fp4_pa_mqa_seg_gluon_kernel(
    const uint8_t* __restrict__ q_p, const uint8_t* __restrict__ q_s,
    const uint8_t* __restrict__ kv, const int* __restrict__ block_tables,
    const float* __restrict__ weights, float* __restrict__ out,
    const int* __restrict__ context_lens,
    int B, int N, int max_model_len, int max_block_len,
    int kv_stride, int split_kv, int block_size, int blk_stride)
{
    constexpr int DP = 64, DS = 4, HT = HEADS / 16, K16 = KVBLK / 16;

    const int pid = blockIdx.x;
    const int nn  = pid % N;
    int rem       = pid / N;
    const int b   = rem % B;
    const int split = rem / B;

    const int ctx = context_lens[b];
    if (ctx <= 0) return;
    const int total_tiles = (ctx + 15) / 16;
    const int tiles_per_split = (total_tiles + split_kv - 1) / split_kv;
    const int tile_start = split * tiles_per_split;
    const int tile_end   = min(tile_start + tiles_per_split, total_tiles);
    if (tile_start >= tile_end) return;

    const int warp = threadIdx.x / LANES;
    const int lane = threadIdx.x % LANES;
    const int lt   = lane & 15;
    const int lg   = lane >> 4;

    const int qn = (b * N + nn);
    const uint8_t* q_p_base = q_p + (size_t)qn * HEADS * DP;
    const uint8_t* q_s_base = q_s + (size_t)qn * HEADS * DS;
    const float*   w_base   = weights + (size_t)qn * HEADS;

    int4v qreg[HT]; int qscale[HT]; float wreg[HT][4];
#pragma unroll
    for (int ht = 0; ht < HT; ++ht) {
        const int head_in = 16 * ht + lt;
        qreg[ht]   = load16v4(q_p_base + (size_t)head_in * DP + 16 * lg);
        qscale[ht] = (int)q_s_base[(size_t)head_in * DS + lg];
#pragma unroll
        for (int r = 0; r < 4; ++r) wreg[ht][r] = w_base[16 * ht + 4 * lg + r];
    }

    const int limit   = ctx - N + nn;
    const size_t bt_row = (size_t)b * max_block_len;
    float* out_row = out + (size_t)qn * max_model_len;
    const size_t data_off  = (size_t)lt * DP + (size_t)lg * 16;
    const int    scale_col = lt * DS + lg;

    auto gather = [&](int t_idx, int4v& bK, int& sbK) {
        const int logical_blk = t_idx / K16;
        const int rank16      = t_idx - logical_blk * K16;
        const int phys = block_tables[bt_row + logical_blk];
        const uint8_t* base = kv + (size_t)phys * blk_stride;
        bK  = load16v4(base + (size_t)rank16 * (16 * DP) + data_off);
        sbK = (int)base[(size_t)KVBLK * DP + (size_t)rank16 * (16 * DS) + scale_col];
    };

    auto store = [&](int t_idx, float part) {
        part = group_sum4(part, lane);
        const int j = t_idx * 16 + lt;
        if (lane < 16 && j < max_model_len)
            out_row[j] = (j <= limit) ? part : NEG_INF;
    };

    // Explicit double-buffered pipeline (gluon 2-stage analogue): prefetch the
    // next tile's K/scale while the MFMAs of the current tile run, and use
    // sched_group_barrier to interleave those loads with the matrix core.
    const int it0 = tile_start + warp;
    int4v Kc, Kn; int Sc, Sn;
    if (it0 < tile_end) gather(it0, Kc, Sc);
    for (int it = it0; it < tile_end; it += NW) {
        const int nxt = it + NW;
        fp4_setprio<3>();
        if (nxt < tile_end) gather(nxt, Kn, Sn);   // VMEM prefetch (1 int4 + 1 byte)
        const int8v Kop = widen(Kc);
        float part = 0.0f;
#pragma unroll
        for (int ht = 0; ht < HT; ++ht) {
            float4v acc = {0.f, 0.f, 0.f, 0.f};
            acc = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
                widen(qreg[ht]), Kop, acc, 4, 4, 0, qscale[ht], 0, Sc);
#pragma unroll
            for (int r = 0; r < 4; ++r) part += fmaxf(acc[r], 0.0f) * wreg[ht][r];
        }
        // Interleave the prefetch VMEM loads with the HT MFMAs (gluon pattern).
        FP4_SGB_VMEM(1); FP4_SGB_MFMA(2); FP4_SGB_VMEM(1); FP4_SGB_MFMA(2);
        fp4_setprio<0>();
        store(it, part);
        Kc = Kn; Sc = Sn;
    }
}

// ---- next_n-fused kernel -------------------------------------------------
// Same math, but one program serves ALL next_n queries of a request: each K
// tile is gathered ONCE (the dominant memory cost) and reused across the N
// MFMAs, cutting KV traffic ~N x. grid = B * split_kv (no N), so the grid is
// still a pure function of batch size -> CUDA-graph stable.
template <int HEADS, int NEXT_N>
__global__ void fp4_pa_mqa_fused_kernel(
    const uint8_t* __restrict__ q_p, const uint8_t* __restrict__ q_s,
    const uint8_t* __restrict__ kv, const int* __restrict__ block_tables,
    const float* __restrict__ weights, float* __restrict__ out,
    const int* __restrict__ context_lens,
    int B, int max_model_len, int max_block_len, int kv_stride, int split_kv)
{
    constexpr int DP = 64, DS = 4, HT = HEADS / 16;

    const int pid = blockIdx.x;
    const int b = pid % B;
    const int split = pid / B;

    const int ctx = context_lens[b];
    if (ctx <= 0) return;

    const int total_tiles = (ctx + 15) / 16;
    const int tiles_per_split = (total_tiles + split_kv - 1) / split_kv;
    const int tile_start = split * tiles_per_split;
    const int tile_end = min(tile_start + tiles_per_split, total_tiles);
    if (tile_start >= tile_end) return;

    const int warp = threadIdx.x / LANES;
    const int lane = threadIdx.x % LANES;
    const int lt = lane & 15;
    const int lg = lane >> 4;

    // Fusion needs N copies of Q, which as registers would halve occupancy. So
    // Q + scales + weights for this request (all next_n) are staged in the idle
    // LDS, leaving the main loop barrier-free with a continuous per-warp gather
    // (high HBM duty) at full occupancy.
    constexpr int DV = DP / 16;        // int4v chunks per head row (= 4)
    __shared__ float   w_lds[NEXT_N * HEADS];
    __shared__ int4v   q_lds[NEXT_N * HEADS * DV];   // [nn][head_in][lg]
    __shared__ uint8_t qs_lds[NEXT_N * HEADS * DS];
    for (int i = threadIdx.x; i < NEXT_N * HEADS; i += WARPS * LANES)
        w_lds[i] = weights[(size_t)(b * NEXT_N) * HEADS + i];
    const uint8_t* q_p_b = q_p + (size_t)(b * NEXT_N) * HEADS * DP;
    const uint8_t* q_s_b = q_s + (size_t)(b * NEXT_N) * HEADS * DS;
    for (int i = threadIdx.x; i < NEXT_N * HEADS * DV; i += WARPS * LANES)
        q_lds[i] = load16v4(q_p_b + (size_t)i * 16);
    for (int i = threadIdx.x; i < NEXT_N * HEADS * DS; i += WARPS * LANES)
        qs_lds[i] = q_s_b[i];
    __syncthreads();   // all warps participate (no per-warp early return above)

    const size_t bt_row = (size_t)b * max_block_len;

    auto gather = [&](int t_idx, int4v& bK, int& sbK) {
        const int jj = t_idx * 16 + lt;
        const int blk = (jj < ctx) ? block_tables[bt_row + jj] : 0;
        const uint8_t* krow = kv + (size_t)blk * kv_stride;
        bK = load16v4(krow + 16 * lg);
        sbK = (int)krow[DP + lg];
    };

    auto compute = [&](int t_idx, int4v bK, int sbK) {
        const int8v Kop = widen(bK);
        const int j = t_idx * 16 + lt;
#pragma unroll
        for (int nn = 0; nn < NEXT_N; ++nn) {
            float part = 0.0f;
#pragma unroll
            for (int ht = 0; ht < HT; ++ht) {
                const int head_in = 16 * ht + lt;
                const int4v Qop = q_lds[(nn * HEADS + head_in) * DV + lg];
                const int   qsc = (int)qs_lds[(nn * HEADS + head_in) * DS + lg];
                float4v acc = {0.f, 0.f, 0.f, 0.f};
                acc = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
                    widen(Qop), Kop, acc, 4, 4, 0, qsc, 0, sbK);
#pragma unroll
                for (int r = 0; r < 4; ++r)
                    part += fmaxf(acc[r], 0.0f) * w_lds[nn * HEADS + 16 * ht + 4 * lg + r];
            }
            part = group_sum4(part, lane);
            if (lane < 16 && j < max_model_len) {
                const int limit = ctx - NEXT_N + nn;
                out[(size_t)(b * NEXT_N + nn) * max_model_len + j] =
                    (j <= limit) ? part : NEG_INF;
            }
        }
    };

    // UNROLL-wide, 2-deep pipeline (same as the base kernel).
    const int it0 = tile_start + warp;
    int4v Kc[UNROLL]; int Sc[UNROLL];
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
        const int t = it0 + u * WARPS;
        if (t < tile_end) gather(t, Kc[u], Sc[u]);
    }
    for (int it = it0; it < tile_end; it += UNROLL * WARPS) {
        int4v Kn[UNROLL]; int Sn[UNROLL];
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const int t = it + (UNROLL + u) * WARPS;
            if (t < tile_end) gather(t, Kn[u], Sn[u]);
        }
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const int t = it + u * WARPS;
            if (t < tile_end) compute(t, Kc[u], Sc[u]);
        }
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) { Kc[u] = Kn[u]; Sc[u] = Sn[u]; }
    }
}

// ---- LDS-K-staged kernel -------------------------------------------------
// The register-fused kernel halves KV traffic but needs N copies of Q in
// registers, which halves occupancy -> it never reaches the gather BW. This
// variant instead stages each K tile in LDS: all warps cooperatively gather a
// chunk of WARPS tiles ONCE from HBM, then the work is split so each warp keeps
// only ONE next_n's Q (low VGPR -> full occupancy) and reads the shared K from
// LDS. HBM traffic is 1x (like fusion) but occupancy stays at base levels.
template <int HEADS, int NEXT_N, int PRIO = 0, bool PERMLANE = true>
__global__ void fp4_pa_mqa_ldsk_kernel(
    const uint8_t* __restrict__ q_p, const uint8_t* __restrict__ q_s,
    const uint8_t* __restrict__ kv, const int* __restrict__ block_tables,
    const float* __restrict__ weights, float* __restrict__ out,
    const int* __restrict__ context_lens,
    int B, int max_model_len, int max_block_len, int kv_stride, int split_kv)
{
    constexpr int DP = 64, DS = 4, HT = HEADS / 16;
    static_assert(WARPS % NEXT_N == 0, "WARPS must be divisible by NEXT_N");
    constexpr int G = WARPS / NEXT_N;     // warps per next_n
    constexpr int CT = TPW * WARPS;       // tiles staged in LDS per chunk

    const int pid = blockIdx.x;
    const int b = pid % B;
    const int split = pid / B;

    const int ctx = context_lens[b];
    if (ctx <= 0) return;

    const int total_tiles = (ctx + 15) / 16;
    const int tiles_per_split = (total_tiles + split_kv - 1) / split_kv;
    const int tile_start = split * tiles_per_split;
    const int tile_end = min(tile_start + tiles_per_split, total_tiles);
    if (tile_start >= tile_end) return;

    const int warp = threadIdx.x / LANES;
    const int lane = threadIdx.x % LANES;
    const int lt = lane & 15;
    const int lg = lane >> 4;

    __shared__ int4v k_lds[CT * LANES];        // CT tiles/chunk, 16 B/lane
    __shared__ int   ks_lds[CT * LANES];       // K scale per lane
    __shared__ float w_lds[NEXT_N * HEADS];    // weights (idle-LDS staged)
#pragma unroll
    for (int i = threadIdx.x; i < NEXT_N * HEADS; i += WARPS * LANES)
        w_lds[i] = weights[(size_t)(b * NEXT_N) * HEADS + i];

    // This warp computes next_n = my_nn for its share of each chunk's tiles.
    const int my_nn = warp / G;
    const int my_off = warp % G;
    const int qn = b * NEXT_N + my_nn;
    const uint8_t* q_p_base = q_p + (size_t)qn * HEADS * DP;
    const uint8_t* q_s_base = q_s + (size_t)qn * HEADS * DS;
    int4v qreg[HT];
    int   qscale[HT];
#pragma unroll
    for (int ht = 0; ht < HT; ++ht) {
        const int head_in = 16 * ht + lt;
        qreg[ht]   = load16v4(q_p_base + (size_t)head_in * DP + 16 * lg);
        qscale[ht] = (int)q_s_base[(size_t)head_in * DS + lg];
    }

    const size_t bt_row = (size_t)b * max_block_len;
    const int limit = ctx - NEXT_N + my_nn;
    float* out_row = out + (size_t)qn * max_model_len;

    // Register-staged gather: each warp gathers TPW tiles per chunk (slot
    // k*WARPS+warp -> tile chunk + k*WARPS + warp). Keeping the next chunk's K
    // in registers (g/gs) overlaps its HBM loads with the current chunk's MFMAs,
    // and the wide chunk gives enough independent tiles between barriers to hide
    // MFMA + reduction latency.
    auto gather = [&](int chunk, int4v g[TPW], int gs[TPW]) {
#pragma unroll
        for (int k = 0; k < TPW; ++k) {
            const int Tg = chunk + k * WARPS + warp;
            const int jj = Tg * 16 + lt;
            const int blk = (Tg < tile_end && jj < ctx) ? block_tables[bt_row + jj] : 0;
            const uint8_t* krow = kv + (size_t)blk * kv_stride;
            g[k]  = load16v4(krow + 16 * lg);
            gs[k] = (int)krow[DP + lg];
        }
    };

    int4v g[TPW]; int gs[TPW];
    gather(tile_start, g, gs);     // prologue: first chunk in registers

    for (int chunk = tile_start; chunk < tile_end; chunk += CT) {
#pragma unroll
        for (int k = 0; k < TPW; ++k) {       // publish current chunk to LDS
            k_lds[(k * WARPS + warp) * LANES + lane]  = g[k];
            ks_lds[(k * WARPS + warp) * LANES + lane] = gs[k];
        }
        __syncthreads();

        const int nchunk = chunk + CT;
        // s_setprio: raise this wave's issue priority while it fires off the next
        // chunk's HBM gather (so loads launch promptly), then drop to 0 for the
        // long MFMA compute so memory-issuing waves on the SIMD interleave.
        if constexpr (PRIO > 0) __builtin_amdgcn_s_setprio(PRIO);
        if (nchunk < tile_end)
            gather(nchunk, g, gs);   // issue next chunk's loads (overlap compute)
        if constexpr (PRIO > 0) __builtin_amdgcn_s_setprio(0);

        // compute current chunk from LDS, reusing each K across this warp's next_n
#pragma unroll
        for (int s = my_off; s < CT; s += G) {
            const int T = chunk + s;
            if (T >= tile_end) continue;
            const int8v Kop = widen(k_lds[s * LANES + lane]);
            const int   sbK = ks_lds[s * LANES + lane];
            float part = 0.0f;
#pragma unroll
            for (int ht = 0; ht < HT; ++ht) {
                float4v acc = {0.f, 0.f, 0.f, 0.f};
                acc = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
                    widen(qreg[ht]), Kop, acc, 4, 4, 0, qscale[ht], 0, sbK);
#pragma unroll
                for (int r = 0; r < 4; ++r)
                    part += fmaxf(acc[r], 0.0f) * w_lds[my_nn * HEADS + 16 * ht + 4 * lg + r];
            }
            // ldsk default: permlane swap keeps the reduction off the (busy) ds unit;
            // PERMLANE=false falls back to ds_bpermute (__shfl_xor) for ablation.
            if constexpr (PERMLANE) part = group_sum4_valu(part, lane);
            else                    part = group_sum4(part, lane);
            const int j = T * 16 + lt;
            if (lane < 16 && j < max_model_len)
                out_row[j] = (j <= limit) ? part : NEG_INF;
        }
        __syncthreads();   // K LDS reused next chunk
    }
}

// ---- Load-balanced LDS-K kernel ------------------------------------------
// ldsk gives every request `split_kv` splits, so at high context-length variance
// long-request blocks run a tail while short-request blocks idle (measured up to
// 1.38x slowdown, tests/loadbal.py). This variant instead flattens all
// (request,tile) work and hands each of the gridDim.x blocks an equal 1/grid
// slice. Grid is still a pure function of batch size (graph-stable); each block
// runs the >=1 request-segments its slice covers with the same pipeline.
template <int HEADS, int NEXT_N, int PRIO = 0, bool PERMLANE = true>
__global__ void fp4_pa_mqa_ldsk_bal_kernel(
    const uint8_t* __restrict__ q_p, const uint8_t* __restrict__ q_s,
    const uint8_t* __restrict__ kv, const int* __restrict__ block_tables,
    const float* __restrict__ weights, float* __restrict__ out,
    const int* __restrict__ context_lens,
    int B, int max_model_len, int max_block_len, int kv_stride, int split_kv)
{
    constexpr int DP = 64, DS = 4, HT = HEADS / 16;
    static_assert(WARPS % NEXT_N == 0, "WARPS must be divisible by NEXT_N");
    constexpr int G = WARPS / NEXT_N;
    constexpr int CT = TPW * WARPS;

    const int warp = threadIdx.x / LANES;
    const int lane = threadIdx.x % LANES;
    const int lt = lane & 15;
    const int lg = lane >> 4;
    const int my_nn = warp / G;
    const int my_off = warp % G;

    __shared__ int4v k_lds[CT * LANES];
    __shared__ int   ks_lds[CT * LANES];
    __shared__ float w_lds[NEXT_N * HEADS];

    int Tall = 0;
#pragma unroll 1
    for (int bb = 0; bb < B; ++bb) {
        const int c = context_lens[bb];
        if (c > 0) Tall += (c + 15) / 16;
    }
    if (Tall == 0) return;
    // Even split of [0,Tall) across gridDim blocks with 32-bit math only.
    // Tall <= B*tiles always fits in int; this avoids the 64-bit divide-by-
    // runtime (gridDim) that inflated VGPR (92->) and cut occupancy. First
    // `rem` blocks get q+1 tiles, rest get q -> identical max-load balance and
    // a gapless partition, so the output is still bit-exact.
    const int Gd = gridDim.x;
    const int q = Tall / Gd, rem = Tall - q * Gd;
    const int bx = blockIdx.x;
    const int lo = bx * q + (bx < rem ? bx : rem);
    const int hi = lo + q + (bx < rem ? 1 : 0);
    if (lo >= hi) return;

    // process tiles [t0,t1) of request b (one segment) with the LDS-K pipeline.
    auto process = [&](int b, int ctx, int t0, int t1) {
        const int qn = b * NEXT_N + my_nn;
        const uint8_t* q_p_base = q_p + (size_t)qn * HEADS * DP;
        const uint8_t* q_s_base = q_s + (size_t)qn * HEADS * DS;
        int4v qreg[HT]; int qscale[HT];
#pragma unroll
        for (int ht = 0; ht < HT; ++ht) {
            const int head_in = 16 * ht + lt;
            qreg[ht] = load16v4(q_p_base + (size_t)head_in * DP + 16 * lg);
            qscale[ht] = (int)q_s_base[(size_t)head_in * DS + lg];
        }
#pragma unroll
        for (int i = threadIdx.x; i < NEXT_N * HEADS; i += WARPS * LANES)
            w_lds[i] = weights[(size_t)(b * NEXT_N) * HEADS + i];

        const size_t bt_row = (size_t)b * max_block_len;
        const int limit = ctx - NEXT_N + my_nn;
        float* out_row = out + (size_t)qn * max_model_len;

        auto gather = [&](int chunk, int4v g[TPW], int gs[TPW]) {
#pragma unroll
            for (int k = 0; k < TPW; ++k) {
                const int Tg = chunk + k * WARPS + warp;
                const int jj = Tg * 16 + lt;
                const int blk = (Tg < t1 && jj < ctx) ? block_tables[bt_row + jj] : 0;
                const uint8_t* krow = kv + (size_t)blk * kv_stride;
                g[k] = load16v4(krow + 16 * lg);
                gs[k] = (int)krow[DP + lg];
            }
        };

        int4v g[TPW]; int gs[TPW];
        gather(t0, g, gs);
        for (int chunk = t0; chunk < t1; chunk += CT) {
#pragma unroll
            for (int k = 0; k < TPW; ++k) {
                k_lds[(k * WARPS + warp) * LANES + lane] = g[k];
                ks_lds[(k * WARPS + warp) * LANES + lane] = gs[k];
            }
            __syncthreads();
            const int nchunk = chunk + CT;
            if constexpr (PRIO > 0) __builtin_amdgcn_s_setprio(PRIO);
            if (nchunk < t1) gather(nchunk, g, gs);
            if constexpr (PRIO > 0) __builtin_amdgcn_s_setprio(0);
#pragma unroll
            for (int s = my_off; s < CT; s += G) {
                const int T = chunk + s;
                if (T >= t1) continue;
                const int8v Kop = widen(k_lds[s * LANES + lane]);
                const int sbK = ks_lds[s * LANES + lane];
                float part = 0.0f;
#pragma unroll
                for (int ht = 0; ht < HT; ++ht) {
                    float4v acc = {0.f, 0.f, 0.f, 0.f};
                    acc = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
                        widen(qreg[ht]), Kop, acc, 4, 4, 0, qscale[ht], 0, sbK);
#pragma unroll
                    for (int r = 0; r < 4; ++r)
                        part += fmaxf(acc[r], 0.0f) * w_lds[my_nn * HEADS + 16 * ht + 4 * lg + r];
                }
                if constexpr (PERMLANE) part = group_sum4_valu(part, lane);
                else                    part = group_sum4(part, lane);
                const int j = T * 16 + lt;
                if (lane < 16 && j < max_model_len)
                    out_row[j] = (j <= limit) ? part : NEG_INF;
            }
            __syncthreads();
        }
    };

    int pref = 0;
#pragma unroll 1
    for (int bb = 0; bb < B; ++bb) {
        const int c = context_lens[bb];
        if (c <= 0) continue;
        const int tb = (c + 15) / 16;
        const int rlo = pref; pref += tb;
        const int slo = lo > rlo ? lo : rlo;
        const int rhi = rlo + tb;
        const int shi = hi < rhi ? hi : rhi;
        if (slo < shi) process(bb, c, slo - rlo, shi - rlo);
    }
}

extern "C" void launch_fp4_pa_mqa_ldsk(
    const uint8_t* q_p, const uint8_t* q_s, const uint8_t* kv,
    const int* block_tables, const float* weights, float* out,
    const int* context_lens,
    int B, int N, int H, int max_model_len, int max_block_len,
    int kv_stride, int split_kv, long stream)
{
    dim3 grid(B * split_kv);
    dim3 block(WARPS * LANES);
    hipStream_t s = reinterpret_cast<hipStream_t>(stream);
    auto launch = [&](auto kern) {
        kern<<<grid, block, 0, s>>>(
            q_p, q_s, kv, block_tables, weights, out, context_lens,
            B, max_model_len, max_block_len, kv_stride, split_kv);
    };
    // PRIO=3: s_setprio around the gather-issue / MFMA boundary. Bit-exact and
    // CUDA-graph-stable; ~5-8% at short context (latency-bound), neutral when
    // BW-bound at long context. See tests/prio_ab.py.
    if (H == 64 && N == 2) launch(fp4_pa_mqa_ldsk_kernel<64, 2, 3>);
    else if (H == 64 && N == 1) launch(fp4_pa_mqa_ldsk_kernel<64, 1, 3>);
    else if (H == 32 && N == 2) launch(fp4_pa_mqa_ldsk_kernel<32, 2, 3>);
    else if (H == 32 && N == 1) launch(fp4_pa_mqa_ldsk_kernel<32, 1, 3>);
    else if (H == 128 && N == 2) launch(fp4_pa_mqa_ldsk_kernel<128, 2, 3>);
    else if (H == 128 && N == 1) launch(fp4_pa_mqa_ldsk_kernel<128, 1, 3>);
}

// Load-balanced ldsk launcher (grid = B*split_kv, same graph-stable grid as ldsk;
// the kernel just redistributes the work evenly across those blocks).
extern "C" void launch_fp4_pa_mqa_ldsk_bal(
    const uint8_t* q_p, const uint8_t* q_s, const uint8_t* kv,
    const int* block_tables, const float* weights, float* out,
    const int* context_lens,
    int B, int N, int H, int max_model_len, int max_block_len,
    int kv_stride, int split_kv, long stream)
{
    dim3 grid(B * split_kv);
    dim3 block(WARPS * LANES);
    hipStream_t s = reinterpret_cast<hipStream_t>(stream);
    auto launch = [&](auto kern) {
        kern<<<grid, block, 0, s>>>(
            q_p, q_s, kv, block_tables, weights, out, context_lens,
            B, max_model_len, max_block_len, kv_stride, split_kv);
    };
    if (H == 64 && N == 2) launch(fp4_pa_mqa_ldsk_bal_kernel<64, 2, 3>);
    else if (H == 64 && N == 1) launch(fp4_pa_mqa_ldsk_bal_kernel<64, 1, 3>);
    else if (H == 32 && N == 2) launch(fp4_pa_mqa_ldsk_bal_kernel<32, 2, 3>);
    else if (H == 32 && N == 1) launch(fp4_pa_mqa_ldsk_bal_kernel<32, 1, 3>);
    else if (H == 128 && N == 2) launch(fp4_pa_mqa_ldsk_bal_kernel<128, 2, 3>);
    else if (H == 128 && N == 1) launch(fp4_pa_mqa_ldsk_bal_kernel<128, 1, 3>);
}

// ldsk + s_setprio experiment. Priority level read from env LDSK_PRIO (1..3).
extern "C" void launch_fp4_pa_mqa_ldsk_prio(
    const uint8_t* q_p, const uint8_t* q_s, const uint8_t* kv,
    const int* block_tables, const float* weights, float* out,
    const int* context_lens,
    int B, int N, int H, int max_model_len, int max_block_len,
    int kv_stride, int split_kv, long stream)
{
    dim3 grid(B * split_kv);
    dim3 block(WARPS * LANES);
    hipStream_t s = reinterpret_cast<hipStream_t>(stream);
    const char* e = getenv("LDSK_PRIO");
    int prio = e ? atoi(e) : 1;
    auto launch = [&](auto kern) {
        kern<<<grid, block, 0, s>>>(
            q_p, q_s, kv, block_tables, weights, out, context_lens,
            B, max_model_len, max_block_len, kv_stride, split_kv);
    };
    if (H == 64 && N == 2) {
        if (prio >= 3) launch(fp4_pa_mqa_ldsk_kernel<64, 2, 3>);
        else if (prio == 2) launch(fp4_pa_mqa_ldsk_kernel<64, 2, 2>);
        else launch(fp4_pa_mqa_ldsk_kernel<64, 2, 1>);
    } else if (H == 64 && N == 1) {
        if (prio >= 3) launch(fp4_pa_mqa_ldsk_kernel<64, 1, 3>);
        else if (prio == 2) launch(fp4_pa_mqa_ldsk_kernel<64, 1, 2>);
        else launch(fp4_pa_mqa_ldsk_kernel<64, 1, 1>);
    } else if (H == 128 && N == 2) {
        if (prio >= 3) launch(fp4_pa_mqa_ldsk_kernel<128, 2, 3>);
        else if (prio == 2) launch(fp4_pa_mqa_ldsk_kernel<128, 2, 2>);
        else launch(fp4_pa_mqa_ldsk_kernel<128, 2, 1>);
    }
}

// ldsk ablation harness (H=64,N=2): PRIO from env LDSK_PRIO (0/3), reduction
// from LDSK_PERMLANE (1=permlane swap / 0=ds_bpermute shuffle). For isolating
// the per-optimization deltas in tests/ablation.py.
extern "C" void launch_fp4_pa_mqa_ldsk_ablate(
    const uint8_t* q_p, const uint8_t* q_s, const uint8_t* kv,
    const int* block_tables, const float* weights, float* out,
    const int* context_lens,
    int B, int N, int H, int max_model_len, int max_block_len,
    int kv_stride, int split_kv, long stream)
{
    dim3 grid(B * split_kv);
    dim3 block(WARPS * LANES);
    hipStream_t s = reinterpret_cast<hipStream_t>(stream);
    const char* ep = getenv("LDSK_PRIO");
    const char* el = getenv("LDSK_PERMLANE");
    int prio = ep ? atoi(ep) : 0;
    int pl = el ? atoi(el) : 1;
    auto launch = [&](auto kern) {
        kern<<<grid, block, 0, s>>>(
            q_p, q_s, kv, block_tables, weights, out, context_lens,
            B, max_model_len, max_block_len, kv_stride, split_kv);
    };
    if (H == 64 && N == 2) {
        if (prio >= 3 && pl)        launch(fp4_pa_mqa_ldsk_kernel<64, 2, 3, true>);
        else if (prio >= 3 && !pl)  launch(fp4_pa_mqa_ldsk_kernel<64, 2, 3, false>);
        else if (prio == 0 && pl)   launch(fp4_pa_mqa_ldsk_kernel<64, 2, 0, true>);
        else                        launch(fp4_pa_mqa_ldsk_kernel<64, 2, 0, false>);
    }
}

extern "C" void launch_fp4_pa_mqa_fused(
    const uint8_t* q_p, const uint8_t* q_s, const uint8_t* kv,
    const int* block_tables, const float* weights, float* out,
    const int* context_lens,
    int B, int N, int H, int max_model_len, int max_block_len,
    int kv_stride, int split_kv, long stream)
{
    dim3 grid(B * split_kv);
    dim3 block(WARPS * LANES);
    hipStream_t s = reinterpret_cast<hipStream_t>(stream);
    auto launch = [&](auto kern) {
        kern<<<grid, block, 0, s>>>(
            q_p, q_s, kv, block_tables, weights, out, context_lens,
            B, max_model_len, max_block_len, kv_stride, split_kv);
    };
    if (H == 64 && N == 2) launch(fp4_pa_mqa_fused_kernel<64, 2>);
    else if (H == 64 && N == 1) launch(fp4_pa_mqa_fused_kernel<64, 1>);
    else if (H == 32 && N == 2) launch(fp4_pa_mqa_fused_kernel<32, 2>);
    else if (H == 32 && N == 1) launch(fp4_pa_mqa_fused_kernel<32, 1>);
    else if (H == 128 && N == 2) launch(fp4_pa_mqa_fused_kernel<128, 2>);
    else if (H == 128 && N == 1) launch(fp4_pa_mqa_fused_kernel<128, 1>);
}

// layout: 0 = interleaved (non-preshuffle, per-token fused cache; KVBlockSize==1),
//         1 = segregated  (non-preshuffle paged, block_size tokens/physical block),
//         2 = preshuffle  (FP8-gluon preshuffle layout, block_size multiple of 16).
// block_size / blk_stride are only consulted for layout 1 and 2; blk_stride is the
// per-physical-block byte stride (= kv_cache.stride(0), vLLM page padding included).
extern "C" void launch_fp4_pa_mqa(
    const uint8_t* q_p, const uint8_t* q_s, const uint8_t* kv,
    const int* block_tables, const float* weights, float* out,
    const int* context_lens,
    int B, int N, int H, int max_model_len, int max_block_len,
    int kv_stride, int split_kv, int layout, int block_size, int blk_stride,
    long stream)
{
    dim3 grid(B * N * split_kv);
    dim3 block(WARPS * LANES);
    hipStream_t s = reinterpret_cast<hipStream_t>(stream);
    auto launch = [&](auto kern) {
        kern<<<grid, block, 0, s>>>(
            q_p, q_s, kv, block_tables, weights, out, context_lens,
            B, N, max_model_len, max_block_len, kv_stride, split_kv,
            block_size, blk_stride);
    };
    if (layout == LAYOUT_PRESHUFFLE) {
        // Dedicated preshuffle kernel for the common compile-time block sizes;
        // fall back to the generic preshuffle branch for other block sizes.
        // UNR (pipeline depth = # of outstanding KV loads per warp) is the main
        // knob for hitting HBM peak in this latency-bound decode kernel; it is
        // overridable via FP4_PS_UNROLL for tuning.
        const char* eu = getenv("FP4_PS_UNROLL");
        const char* ew = getenv("FP4_PS_WARPS");
        const char* eps = getenv("FP4_PS_SCHED");  // gluon-style MFMA s_setprio hint
        const bool ps_sched = eps ? atoi(eps) != 0 : true;  // tuned: on by default
        const int unr = eu ? atoi(eu) : 2;   // pipeline depth (regs); 2 is best
        // 8 warps/CTA maximizes memory-level parallelism and is the best choice
        // once each CTA has enough tiles to keep them busy; for small per-CTA
        // tile counts the extra warps just add tail/overhead, so fall back to 4.
        const long tiles_total = (long)max_block_len * block_size / 16;
        const int tiles_per_cta = (int)(tiles_total / (split_kv > 0 ? split_kv : 1));
        const int nw = ew ? atoi(ew) : (tiles_per_cta >= 32 ? 8 : 4);
        auto launch_ps = [&](auto kern, int nwarps) {
            dim3 blk(nwarps * LANES);
            kern<<<grid, blk, 0, s>>>(
                q_p, q_s, kv, block_tables, weights, out, context_lens,
                B, N, max_model_len, max_block_len, kv_stride, split_kv,
                block_size, blk_stride);
        };
#define FP4_PS_BY_H(BS, U, W, S)                                                    \
        do {                                                                       \
            if (H == 64) launch_ps(fp4_pa_mqa_preshuffle_kernel<64, BS, U, W, S>, W); \
            else if (H == 32) launch_ps(fp4_pa_mqa_preshuffle_kernel<32, BS, U, W, S>, W); \
            else if (H == 128) launch_ps(fp4_pa_mqa_preshuffle_kernel<128, BS, U, W, S>, W); \
        } while (0)
#define FP4_PS_BY_S(BS, U, W)                                                       \
        do {                                                                       \
            if (ps_sched) FP4_PS_BY_H(BS, U, W, true); else FP4_PS_BY_H(BS, U, W, false); \
        } while (0)
#define FP4_PS_BY_U(BS, W)                                                          \
        do {                                                                       \
            if (unr >= 4) FP4_PS_BY_S(BS, 4, W); else FP4_PS_BY_S(BS, 2, W);        \
        } while (0)
#define FP4_PS_DISPATCH(BS)                                                         \
        do {                                                                       \
            if (nw >= 8) FP4_PS_BY_U(BS, 8); else FP4_PS_BY_U(BS, 4);               \
        } while (0)
        if (block_size == 16)      FP4_PS_DISPATCH(16);
        else if (block_size == 32) FP4_PS_DISPATCH(32);
        else if (block_size == 64) FP4_PS_DISPATCH(64);
        else {
            if (H == 64)       launch(fp4_pa_mqa_kernel<64, LAYOUT_PRESHUFFLE>);
            else if (H == 32)  launch(fp4_pa_mqa_kernel<32, LAYOUT_PRESHUFFLE>);
            else if (H == 128) launch(fp4_pa_mqa_kernel<128, LAYOUT_PRESHUFFLE>);
        }
#undef FP4_PS_DISPATCH
#undef FP4_PS_BY_U
#undef FP4_PS_BY_S
#undef FP4_PS_BY_H
    } else if (layout == LAYOUT_SEGREGATED) {
        // Dedicated compile-time-KVBLK segregated kernel (block_size in
        // {16,32,64}); generic fallback otherwise. Removes the per-lane div/mod
        // and the 16 redundant per-lane block_tables loads/tile that made the
        // generic segregated path lose to FP8 gluon. Knobs FP4_SEG_UNROLL /
        // FP4_SEG_WARPS / FP4_SEG_SCHED; defaults (unr=2, nw=4, sched=off) are
        // safe (8 warps regress segregated at large batch).
        const char* eu = getenv("FP4_SEG_UNROLL");
        const char* ew = getenv("FP4_SEG_WARPS");
        const char* es = getenv("FP4_SEG_SCHED");
        // Tuned defaults (tune_indexer.py, gfx950, dedicated seg kernel): 8 warps
        // once a CTA has enough tiles to feed them (else they idle at tiny batch
        // / short ctx), plus the gluon-style s_setprio hint. Unlike the old
        // generic kernel, 8 warps no longer regress at large batch here.
        const long tiles_total   = (long)max_block_len * block_size / 16;
        const int  tiles_per_cta = (int)(tiles_total / (split_kv > 0 ? split_kv : 1));
        const long grid_ctas     = (long)B * N * (split_kv > 0 ? split_kv : 1);
        const int  unr   = eu ? atoi(eu) : 2;
        // 8 warps win when a CTA has plenty of work (tiles_per_cta>=48) OR when
        // the grid is small enough that the extra warps fill the CUs rather than
        // over-subscribe (tiles_per_cta>=24 && grid<=1024). The grid clause is
        // what separates B=16/kv8192 (grid 512 -> 8 warps, wins) from
        // B=64/kv8192 (grid 2048 -> 4 warps, avoids over-subscription regress).
        const int  nw    = ew ? atoi(ew)
                              : (((tiles_per_cta >= 16 && grid_ctas <= 1024)
                                  || tiles_per_cta >= 48) ? 8 : 4);
        // s_setprio + iglp scheduling helps once there is enough work to overlap,
        // but ADDS overhead at the tiniest shapes (e.g. B=4/kv8192 has ~4 tiles
        // per CTA), so gate it on tiles_per_cta.
        const bool sched = es ? atoi(es) != 0 : (tiles_per_cta >= 8);
        auto launch_seg = [&](auto kern, int nwarps) {
            dim3 blk(nwarps * LANES);
            kern<<<grid, blk, 0, s>>>(
                q_p, q_s, kv, block_tables, weights, out, context_lens,
                B, N, max_model_len, max_block_len, kv_stride, split_kv,
                block_size, blk_stride);
        };
        // Opt-in gluon-asm-aligned kernel (explicit pipeline + sched_group_barrier).
        if (getenv("FP4_SEG_GLUON") &&
            (block_size == 16 || block_size == 32 || block_size == 64)) {
            const int gw = nw;
#define FP4_SEGG_BY_H(BS, W)                                                        \
            do {                                                                   \
                if (H == 64) launch_seg(fp4_pa_mqa_seg_gluon_kernel<64, BS, 2, W>, W);  \
                else if (H == 32) launch_seg(fp4_pa_mqa_seg_gluon_kernel<32, BS, 2, W>, W); \
                else if (H == 128) launch_seg(fp4_pa_mqa_seg_gluon_kernel<128, BS, 2, W>, W); \
            } while (0)
#define FP4_SEGG(BS) do { if (gw >= 8) FP4_SEGG_BY_H(BS, 8); else FP4_SEGG_BY_H(BS, 4); } while (0)
            if (block_size == 16)      FP4_SEGG(16);
            else if (block_size == 32) FP4_SEGG(32);
            else                       FP4_SEGG(64);
#undef FP4_SEGG
#undef FP4_SEGG_BY_H
            return;
        }
#define FP4_SEG_BY_H(BS, U, W, S)                                                   \
        do {                                                                       \
            if (H == 64) launch_seg(fp4_pa_mqa_seg_kernel<64, BS, U, W, S>, W);     \
            else if (H == 32) launch_seg(fp4_pa_mqa_seg_kernel<32, BS, U, W, S>, W);\
            else if (H == 128) launch_seg(fp4_pa_mqa_seg_kernel<128, BS, U, W, S>, W);\
        } while (0)
#define FP4_SEG_BY_S(BS, U, W)                                                      \
        do { if (sched) FP4_SEG_BY_H(BS, U, W, true); else FP4_SEG_BY_H(BS, U, W, false); } while (0)
#define FP4_SEG_BY_U(BS, W)                                                         \
        do { if (unr >= 4) FP4_SEG_BY_S(BS, 4, W); else FP4_SEG_BY_S(BS, 2, W); } while (0)
#define FP4_SEG_DISPATCH(BS)                                                        \
        do { if (nw >= 8) FP4_SEG_BY_U(BS, 8); else FP4_SEG_BY_U(BS, 4); } while (0)
        if (block_size == 16)      FP4_SEG_DISPATCH(16);
        else if (block_size == 32) FP4_SEG_DISPATCH(32);
        else if (block_size == 64) FP4_SEG_DISPATCH(64);
        else {  // uncommon block size -> generic runtime-block_size kernel
            if (H == 64)       launch(fp4_pa_mqa_kernel<64, LAYOUT_SEGREGATED>);
            else if (H == 32)  launch(fp4_pa_mqa_kernel<32, LAYOUT_SEGREGATED>);
            else if (H == 128) launch(fp4_pa_mqa_kernel<128, LAYOUT_SEGREGATED>);
        }
#undef FP4_SEG_DISPATCH
#undef FP4_SEG_BY_U
#undef FP4_SEG_BY_S
#undef FP4_SEG_BY_H
    } else {
        if (H == 64)       launch(fp4_pa_mqa_kernel<64, LAYOUT_INTERLEAVED>);
        else if (H == 32)  launch(fp4_pa_mqa_kernel<32, LAYOUT_INTERLEAVED>);
        else if (H == 128) launch(fp4_pa_mqa_kernel<128, LAYOUT_INTERLEAVED>);
    }
}

// ==========================================================================
// Dense (non-paged) MXFP4 MQA-logits  --  prefill counterpart of fp8_mqa_logits
// --------------------------------------------------------------------------
//   logits[m, n] = sum_h relu( Q[m,h,:] . K[n,:] ) * w[m,h]
//   for ks[m] <= n < ke[m], else left untouched (caller pre-fills -inf).
//
// K is a DENSE contiguous tensor [Nkv, D] (MXFP4: kv_p [Nkv,D/2] + kv_s
// [Nkv,D/32]); no block table. Each query row m is independent with its own
// [ks[m],ke[m]) range. Same scaled-FP4 matrix core as the paged kernels.
//
// One program per (row m, split): the WARPS warps split the row's KV tiles and
// stream K straight from HBM (LDS-free -> reduction uses ds_bpermute, which is
// otherwise idle). FP4 reads half the KV bytes of the fp8 path.
// ==========================================================================
template <int HEADS>
__global__ void fp4_mqa_dense_kernel(
    const uint8_t* __restrict__ q_p,        // [M, H, D/2]
    const uint8_t* __restrict__ q_s,        // [M, H, D/32]
    const uint8_t* __restrict__ kv_p,       // [Nkv, D/2]
    const uint8_t* __restrict__ kv_s,       // [Nkv, D/32]
    const float*   __restrict__ weights,    // [M, H]
    const int*     __restrict__ cu_starts,  // [M]
    const int*     __restrict__ cu_ends,    // [M]
    float*         __restrict__ out,        // [M, Nkv]  (pre-filled -inf)
    int M, int Nkv, int kv_p_stride, int kv_s_stride, int out_stride, int split_kv)
{
    constexpr int DP = 64, DS = 4, HT = HEADS / 16;

    const int pid   = blockIdx.x;
    const int m     = pid % M;
    const int split = pid / M;

    const int ks = cu_starts[m];
    const int ke = cu_ends[m];
    if (ks >= ke) return;

    const int t_lo = ks >> 4;             // floor(ks/16)
    const int t_hi = (ke + 15) >> 4;      // ceil(ke/16)
    const int total_tiles = t_hi - t_lo;
    const int tiles_per_split = (total_tiles + split_kv - 1) / split_kv;
    const int tile_start = t_lo + split * tiles_per_split;
    const int tile_end   = min(tile_start + tiles_per_split, t_hi);
    if (tile_start >= tile_end) return;

    const int warp = threadIdx.x / LANES;
    const int lane = threadIdx.x % LANES;
    const int lt   = lane & 15;           // token within tile / Q head row
    const int lg   = lane >> 4;           // 32-elem K group / head-group

    // Q + scales + weights for row m, loaded once, reused over all tiles.
    const uint8_t* q_p_base = q_p + (size_t)m * HEADS * DP;
    const uint8_t* q_s_base = q_s + (size_t)m * HEADS * DS;
    const float*   w_base   = weights + (size_t)m * HEADS;
    int4v qreg[HT];
    int   qscale[HT];
    float wreg[HT][4];
#pragma unroll
    for (int ht = 0; ht < HT; ++ht) {
        const int head_in = 16 * ht + lt;
        qreg[ht]   = load16v4(q_p_base + (size_t)head_in * DP + 16 * lg);
        qscale[ht] = (int)q_s_base[(size_t)head_in * DS + lg];
#pragma unroll
        for (int r = 0; r < 4; ++r)
            wreg[ht][r] = w_base[16 * ht + 4 * lg + r];
    }
    float* out_row = out + (size_t)m * out_stride;

    auto gather = [&](int t_idx, int4v& bK, int& sbK) {
        const int n = t_idx * 16 + lt;
        const int nn = (n < Nkv) ? n : 0;            // clamp OOB read (masked out)
        const uint8_t* kp  = kv_p + (size_t)nn * kv_p_stride;
        const uint8_t* ksc = kv_s + (size_t)nn * kv_s_stride;
        bK  = load16v4(kp + 16 * lg);
        sbK = (int)ksc[lg];
    };

    auto compute = [&](int t_idx, int4v bK, int sbK) {
        const int8v Kop = widen(bK);
        float part = 0.0f;
#pragma unroll
        for (int ht = 0; ht < HT; ++ht) {
            float4v acc = {0.f, 0.f, 0.f, 0.f};
            acc = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
                widen(qreg[ht]), Kop, acc, 4, 4, 0, qscale[ht], 0, sbK);
#pragma unroll
            for (int r = 0; r < 4; ++r)
                part += fmaxf(acc[r], 0.0f) * wreg[ht][r];
        }
        part = group_sum4(part, lane);
        const int n = t_idx * 16 + lt;
        if (lane < 16 && n >= ks && n < ke)
            out_row[n] = part;
    };

    // UNROLL-wide, 2-deep software pipeline, warp-strided tiles (HBM-latency bound).
    const int it0 = tile_start + warp;
    int4v Kc[UNROLL]; int Sc[UNROLL];
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
        const int t = it0 + u * WARPS;
        if (t < tile_end) gather(t, Kc[u], Sc[u]);
    }
    for (int it = it0; it < tile_end; it += UNROLL * WARPS) {
        int4v Kn[UNROLL]; int Sn[UNROLL];
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const int t = it + (UNROLL + u) * WARPS;
            if (t < tile_end) gather(t, Kn[u], Sn[u]);
        }
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const int t = it + u * WARPS;
            if (t < tile_end) compute(t, Kc[u], Sc[u]);
        }
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) { Kc[u] = Kn[u]; Sc[u] = Sn[u]; }
    }
}

extern "C" void launch_fp4_mqa_dense(
    const uint8_t* q_p, const uint8_t* q_s,
    const uint8_t* kv_p, const uint8_t* kv_s,
    const float* weights, const int* cu_starts, const int* cu_ends,
    float* out, int M, int Nkv, int H,
    int kv_p_stride, int kv_s_stride, int out_stride, int split_kv, long stream)
{
    dim3 grid(M * split_kv);
    dim3 block(WARPS * LANES);
    hipStream_t s = reinterpret_cast<hipStream_t>(stream);
    auto launch = [&](auto kern) {
        kern<<<grid, block, 0, s>>>(
            q_p, q_s, kv_p, kv_s, weights, cu_starts, cu_ends, out,
            M, Nkv, kv_p_stride, kv_s_stride, out_stride, split_kv);
    };
    if (H == 64)       launch(fp4_mqa_dense_kernel<64>);
    else if (H == 32)  launch(fp4_mqa_dense_kernel<32>);
    else if (H == 128) launch(fp4_mqa_dense_kernel<128>);
}

// ==========================================================================
// Dense MXFP4 MQA-logits using the 32x32x64 matrix core (CDNA4)
// --------------------------------------------------------------------------
// Same math/interface as fp4_mqa_dense_kernel, but built on
//   v_mfma_scale_f32_32x32x64_f8f6f4  (32 heads x 32 tokens, K=64 fp4).
// A single 32x32x64 op does 2x the MACs of a 16x16x128 op, so the full
// [64 heads x 32 tokens x D=128] tile costs 4 MFMAs instead of 8 -- half the
// matrix-issue rate, which is what bounds this kernel once KV is L2-resident
// (matches aiter's gluon fp8 path that uses the 32x32 core).
//
// Lane layout (reverse-engineered in tests/probe_mfma32.hip, CLEAN=1):
//   operands : lane l -> row/col = l&31, micro-block half hi = l>>5
//   acc[e]   : token j = l&31,  head i = 8*(e>>2) + 4*hi + (e&3)   (e=0..15)
// So lanes j and j+32 hold complementary head-halves of the SAME token j:
// the head reduction is a single ^32 shuffle (vs two for the 16x16 core).
// A and B share the byte/scale picking, so the K split is contraction-order
// invariant -- only consistency between Q and K matters.
template <int HEADS>
__global__ void fp4_mqa_dense32_kernel(
    const uint8_t* __restrict__ q_p,        // [M, H, D/2]
    const uint8_t* __restrict__ q_s,        // [M, H, D/32]
    const uint8_t* __restrict__ kv_p,       // [Nkv, D/2]
    const uint8_t* __restrict__ kv_s,       // [Nkv, D/32]
    const float*   __restrict__ weights,    // [M, H]
    const int*     __restrict__ cu_starts,  // [M]
    const int*     __restrict__ cu_ends,    // [M]
    float*         __restrict__ out,        // [M, Nkv]  (pre-filled -inf)
    int M, int Nkv, int kv_p_stride, int kv_s_stride, int out_stride, int split_kv)
{
    constexpr int DP = 64, DS = 4, HT2 = HEADS / 32;  // 32-head tiles
    constexpr int BN = 32;                            // tokens per tile

    const int pid   = blockIdx.x;
    const int m     = pid % M;
    const int split = pid / M;

    const int ks = cu_starts[m];
    const int ke = cu_ends[m];
    if (ks >= ke) return;

    const int t_lo = ks >> 5;             // floor(ks/32)
    const int t_hi = (ke + 31) >> 5;      // ceil(ke/32)
    const int total_tiles = t_hi - t_lo;
    const int tiles_per_split = (total_tiles + split_kv - 1) / split_kv;
    const int tile_start = t_lo + split * tiles_per_split;
    const int tile_end   = min(tile_start + tiles_per_split, t_hi);
    if (tile_start >= tile_end) return;

    const int warp = threadIdx.x / LANES;
    const int lane = threadIdx.x % LANES;
    const int rc   = lane & 31;           // row(Q head) / col(token) within tile
    const int hi   = lane >> 5;           // micro-block half (0 -> blocks 0,2 ; 1 -> 1,3)

    // Q + scales + weights for row m, loaded once, reused over all tiles.
    const uint8_t* q_p_base = q_p + (size_t)m * HEADS * DP;
    const uint8_t* q_s_base = q_s + (size_t)m * HEADS * DS;
    const float*   w_base   = weights + (size_t)m * HEADS;
    int4v qreg[HT2][2];                   // [head-tile][MFMA k-step]
    int   qscale[HT2][2];
#if defined(FP4_DENSE_PIPE)
    // Pipelined variant: stage the row's HEADS weights in (otherwise-idle) LDS
    // instead of 32 per-lane VGPR. Freeing those registers is what lets the
    // group-pipeline below keep two MFMA accumulators (prev/cur) live so tile
    // u+1's MFMA overlaps tile u's relu-reduction.
    __shared__ float w_lds[HEADS];
    for (int i = threadIdx.x; i < HEADS; i += WARPS * LANES)
        w_lds[i] = w_base[i];
    __syncthreads();
#else
    float wreg[HT2][16];                  // weight per (head-tile, acc slot e)
#endif
#pragma unroll
    for (int ht = 0; ht < HT2; ++ht) {
        const int head = 32 * ht + rc;
        // Each lane owns a CONTIGUOUS 32B half of the row (micro-blocks 2hi,2hi+1):
        // hi=0 -> bytes[0:32], hi=1 -> bytes[32:64]. The two MFMA k-steps then read
        // adjacent 16B halves, so the 32B load coalesces into one wide transaction.
        // A and B use the same split, so the contraction stays correct.
        const uint8_t* qb = q_p_base + (size_t)head * DP + 32 * hi;
        qreg[ht][0]   = load16v4(qb);
        qreg[ht][1]   = load16v4(qb + 16);
        qscale[ht][0] = (int)q_s_base[(size_t)head * DS + 2 * hi];
        qscale[ht][1] = (int)q_s_base[(size_t)head * DS + 2 * hi + 1];
#if !defined(FP4_DENSE_PIPE)
#pragma unroll
        for (int e = 0; e < 16; ++e) {
            const int hd = 32 * ht + 8 * (e >> 2) + 4 * hi + (e & 3);
            wreg[ht][e] = w_base[hd];
        }
#endif
    }
    float* out_row = out + (size_t)m * out_stride;

    auto gather = [&](int t_idx, int4v& b0, int4v& b1, int& s0, int& s1) {
        const int n  = t_idx * BN + rc;
        const int nn = (n < Nkv) ? n : 0;             // clamp OOB read (masked out)
        const uint8_t* kp  = kv_p + (size_t)nn * kv_p_stride + 32 * hi;
        const uint8_t* ksc = kv_s + (size_t)nn * kv_s_stride;
        b0 = load16v4(kp);
        b1 = load16v4(kp + 16);
        s0 = (int)ksc[2 * hi];
        s1 = (int)ksc[2 * hi + 1];
    };

#if defined(FP4_DENSE_PIPE)
    // ---- group-pipelined path: overlap reduce(group u) with MFMA(group u+1) ----
    // Each tile has HT2 MFMA-groups (one per 32-head tile). We issue group g+1's
    // MFMAs, then run group g's relu-reduction while those MFMAs are in flight on
    // the matrix pipe (VALU and matrix pipes run concurrently). Weights come from
    // LDS (w_lds), which is what freed the VGPRs to keep two acc's (prev/cur) live.
    auto mfma_group = [&](int ht, int8v K0, int8v K1, int s0, int s1) -> float16v {
        float16v acc;
#pragma unroll
        for (int e = 0; e < 16; ++e) acc[e] = 0.f;
        acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
            widen(qreg[ht][0]), K0, acc, 4, 4, 0, qscale[ht][0], 0, s0);
        acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
            widen(qreg[ht][1]), K1, acc, 4, 4, 0, qscale[ht][1], 0, s1);
        return acc;
    };
    float pp0 = 0.f, pp1 = 0.f, pp2 = 0.f, pp3 = 0.f;  // current tile's partial
    auto reduce_group = [&](int t_idx, int ht, const float16v& acc) {
        if (ht == 0) { pp0 = pp1 = pp2 = pp3 = 0.f; }
#pragma unroll
        for (int e = 0; e < 16; e += 4) {
            pp0 += fmaxf(acc[e + 0], 0.0f) * w_lds[32 * ht + 8 * (e >> 2) + 4 * hi + 0];
            pp1 += fmaxf(acc[e + 1], 0.0f) * w_lds[32 * ht + 8 * (e >> 2) + 4 * hi + 1];
            pp2 += fmaxf(acc[e + 2], 0.0f) * w_lds[32 * ht + 8 * (e >> 2) + 4 * hi + 2];
            pp3 += fmaxf(acc[e + 3], 0.0f) * w_lds[32 * ht + 8 * (e >> 2) + 4 * hi + 3];
        }
        if (ht == HT2 - 1) {
            float part = (pp0 + pp1) + (pp2 + pp3);
            part += __shfl_xor(part, 32);
            const int n = t_idx * BN + rc;
            if (lane < 32 && n >= ks && n < ke) out_row[n] = part;
        }
    };

    const int it0 = tile_start + warp;
    float16v acc_prev;
    int prev_t = -1, prev_ht = -1;
    int4v Kc0, Kc1; int Sc0, Sc1;
    for (int t = it0; t < tile_end; t += WARPS) {
        gather(t, Kc0, Kc1, Sc0, Sc1);
        const int8v K0 = widen(Kc0), K1 = widen(Kc1);
#pragma unroll
        for (int ht = 0; ht < HT2; ++ht) {
            const float16v acc_cur = mfma_group(ht, K0, K1, Sc0, Sc1);  // matrix pipe
            if (prev_ht >= 0) reduce_group(prev_t, prev_ht, acc_prev);  // VALU overlaps
            acc_prev = acc_cur; prev_t = t; prev_ht = ht;
        }
    }
    if (prev_ht >= 0) reduce_group(prev_t, prev_ht, acc_prev);
#else

    auto compute = [&](int t_idx, int4v b0, int4v b1, int s0, int s1) {
        const int8v K0 = widen(b0), K1 = widen(b1);
        float part = 0.0f;
#if defined(FP4_PROBE) && FP4_PROBE == 1
        // load-only: isolate pure KV read BW (no MFMA, no reduction)
        part = (float)(b0[0]^b0[1]^b0[2]^b0[3]^b1[0]^b1[1]^b1[2]^b1[3]^s0^s1);
#else
        // Reduction over heads: logit = sum_h relu(qk[h]) * w[h]. Profiled as the
        // kernel's dominant exposed cost: the epilogue is 64 VALU/tile (32 v_max
        // relu + 32 fma); the MFMA hides ~half, and the relu (~389us at 8192x32768)
        // is the exposed remainder. gfx950 has no packed-f32 max so relu can't be
        // packed; 4 independent accumulators (FP add is non-associative) keep ILP
        // and avoid a 32-deep serial dependency. (A packed-f32 v_pk_fma variant and
        // 8-chain variant were both measured slightly SLOWER.)
        float p0 = 0.f, p1 = 0.f, p2 = 0.f, p3 = 0.f;
#pragma unroll
        for (int ht = 0; ht < HT2; ++ht) {
            float16v acc;
#pragma unroll
            for (int e = 0; e < 16; ++e) acc[e] = 0.f;
            acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
                widen(qreg[ht][0]), K0, acc, 4, 4, 0, qscale[ht][0], 0, s0);
            acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
                widen(qreg[ht][1]), K1, acc, 4, 4, 0, qscale[ht][1], 0, s1);
#if defined(FP4_PROBE) && FP4_PROBE == 2
#pragma unroll
            for (int e = 0; e < 16; ++e) p0 += acc[e];     // MFMA, no relu/weight
#else
#pragma unroll
            for (int e = 0; e < 16; e += 4) {
                p0 += fmaxf(acc[e + 0], 0.0f) * wreg[ht][e + 0];
                p1 += fmaxf(acc[e + 1], 0.0f) * wreg[ht][e + 1];
                p2 += fmaxf(acc[e + 2], 0.0f) * wreg[ht][e + 2];
                p3 += fmaxf(acc[e + 3], 0.0f) * wreg[ht][e + 3];
            }
#endif
        }
        part = (p0 + p1) + (p2 + p3);
#endif
        part += __shfl_xor(part, 32);     // lanes j and j+32 -> full 64-head sum
        const int n = t_idx * BN + rc;
        if (lane < 32 && n >= ks && n < ke)
            out_row[n] = part;
    };

    // UNROLL-wide, 2-deep software pipeline, warp-strided tiles.
    const int it0 = tile_start + warp;
    int4v Kc0[UNROLL], Kc1[UNROLL]; int Sc0[UNROLL], Sc1[UNROLL];
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
        const int t = it0 + u * WARPS;
        if (t < tile_end) gather(t, Kc0[u], Kc1[u], Sc0[u], Sc1[u]);
    }
    for (int it = it0; it < tile_end; it += UNROLL * WARPS) {
        int4v Kn0[UNROLL], Kn1[UNROLL]; int Sn0[UNROLL], Sn1[UNROLL];
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const int t = it + (UNROLL + u) * WARPS;
            if (t < tile_end) gather(t, Kn0[u], Kn1[u], Sn0[u], Sn1[u]);
        }
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const int t = it + u * WARPS;
            if (t < tile_end) compute(t, Kc0[u], Kc1[u], Sc0[u], Sc1[u]);
        }
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            Kc0[u] = Kn0[u]; Kc1[u] = Kn1[u]; Sc0[u] = Sn0[u]; Sc1[u] = Sn1[u];
        }
    }
#endif  // FP4_DENSE_PIPE
}

// ==========================================================================
// M-tiled dense MXFP4 MQA-logits, 32x32x64 core (CDNA4)
// --------------------------------------------------------------------------
// In the dominant dense case (M=2048, Nkv=8192) every one of the M blocks of
// fp4_mqa_dense32_kernel independently streams the ENTIRE KV (Nkv*68 bytes).
// KV is identical across all M rows, so that traffic is M-fold redundant and
// is the binding cost once KV no longer fits in L2.
//
// This variant processes MROWS consecutive m-rows per block: each K tile is
// gathered ONCE (shared registers Kc0/Kc1) and fed into MROWS independent MFMA
// reductions, cutting KV global traffic by MROWS x. Q/weights for the MROWS
// rows live in registers (loaded once, reused over all tiles); the K registers
// stay shared, so register growth is only in the (small) per-row Q/acc state.
template <int HEADS, int MROWS>
__global__ void fp4_mqa_dense32_mtile_kernel(
    const uint8_t* __restrict__ q_p,        // [M, H, D/2]
    const uint8_t* __restrict__ q_s,        // [M, H, D/32]
    const uint8_t* __restrict__ kv_p,       // [Nkv, D/2]
    const uint8_t* __restrict__ kv_s,       // [Nkv, D/32]
    const float*   __restrict__ weights,    // [M, H]
    const int*     __restrict__ cu_starts,  // [M]
    const int*     __restrict__ cu_ends,    // [M]
    float*         __restrict__ out,        // [M, Nkv]  (pre-filled -inf)
    int M, int Nkv, int kv_p_stride, int kv_s_stride, int out_stride, int split_kv)
{
    constexpr int DP = 64, DS = 4, HT2 = HEADS / 32;  // 32-head tiles
    constexpr int BN = 32;                            // tokens per tile

    const int num_mb = (M + MROWS - 1) / MROWS;
    const int pid    = blockIdx.x;
    const int mb     = pid % num_mb;
    const int split  = pid / num_mb;
    const int m_base = mb * MROWS;
    const int nrows  = min(MROWS, M - m_base);
    if (nrows <= 0) return;

    // Union tile range over the block's rows (per-row store mask keeps it exact).
    int ks_min = cu_starts[m_base], ke_max = cu_ends[m_base];
#pragma unroll
    for (int r = 1; r < MROWS; ++r) {
        if (r < nrows) {
            ks_min = min(ks_min, cu_starts[m_base + r]);
            ke_max = max(ke_max, cu_ends[m_base + r]);
        }
    }
    if (ks_min >= ke_max) return;

    const int t_lo = ks_min >> 5;
    const int t_hi = (ke_max + 31) >> 5;
    const int total_tiles = t_hi - t_lo;
    const int tiles_per_split = (total_tiles + split_kv - 1) / split_kv;
    const int tile_start = t_lo + split * tiles_per_split;
    const int tile_end   = min(tile_start + tiles_per_split, t_hi);
    if (tile_start >= tile_end) return;

    const int warp = threadIdx.x / LANES;
    const int lane = threadIdx.x % LANES;
    const int rc   = lane & 31;
    const int hi   = lane >> 5;

    // Q + scales + weights for the MROWS rows, loaded once, reused over tiles.
    int4v qreg[MROWS][HT2][2];
    int   qscale[MROWS][HT2][2];
    float wreg[MROWS][HT2][16];
    int   ks_r[MROWS], ke_r[MROWS];
#pragma unroll
    for (int r = 0; r < MROWS; ++r) {
        const int m = (r < nrows) ? (m_base + r) : m_base;
        ks_r[r] = cu_starts[m];
        ke_r[r] = cu_ends[m];
        const uint8_t* q_p_base = q_p + (size_t)m * HEADS * DP;
        const uint8_t* q_s_base = q_s + (size_t)m * HEADS * DS;
        const float*   w_base   = weights + (size_t)m * HEADS;
#pragma unroll
        for (int ht = 0; ht < HT2; ++ht) {
            const int head = 32 * ht + rc;
            const uint8_t* qb = q_p_base + (size_t)head * DP + 32 * hi;
            qreg[r][ht][0]   = load16v4(qb);
            qreg[r][ht][1]   = load16v4(qb + 16);
            qscale[r][ht][0] = (int)q_s_base[(size_t)head * DS + 2 * hi];
            qscale[r][ht][1] = (int)q_s_base[(size_t)head * DS + 2 * hi + 1];
#pragma unroll
            for (int e = 0; e < 16; ++e) {
                const int hd = 32 * ht + 8 * (e >> 2) + 4 * hi + (e & 3);
                wreg[r][ht][e] = w_base[hd];
            }
        }
    }

    auto gather = [&](int t_idx, int4v& b0, int4v& b1, int& s0, int& s1) {
        const int n  = t_idx * BN + rc;
        const int nn = (n < Nkv) ? n : 0;
        const uint8_t* kp  = kv_p + (size_t)nn * kv_p_stride + 32 * hi;
        const uint8_t* ksc = kv_s + (size_t)nn * kv_s_stride;
        b0 = load16v4(kp);
        b1 = load16v4(kp + 16);
        s0 = (int)ksc[2 * hi];
        s1 = (int)ksc[2 * hi + 1];
    };

    auto compute = [&](int t_idx, int4v kb0, int4v kb1, int s0, int s1) {
        const int8v K0 = widen(kb0), K1 = widen(kb1);
        const int n = t_idx * BN + rc;
#pragma unroll
        for (int r = 0; r < MROWS; ++r) {
            float p0 = 0.f, p1 = 0.f, p2 = 0.f, p3 = 0.f;
#pragma unroll
            for (int ht = 0; ht < HT2; ++ht) {
                float16v acc;
#pragma unroll
                for (int e = 0; e < 16; ++e) acc[e] = 0.f;
                acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
                    widen(qreg[r][ht][0]), K0, acc, 4, 4, 0, qscale[r][ht][0], 0, s0);
                acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
                    widen(qreg[r][ht][1]), K1, acc, 4, 4, 0, qscale[r][ht][1], 0, s1);
#pragma unroll
                for (int e = 0; e < 16; e += 4) {
                    p0 += fmaxf(acc[e + 0], 0.0f) * wreg[r][ht][e + 0];
                    p1 += fmaxf(acc[e + 1], 0.0f) * wreg[r][ht][e + 1];
                    p2 += fmaxf(acc[e + 2], 0.0f) * wreg[r][ht][e + 2];
                    p3 += fmaxf(acc[e + 3], 0.0f) * wreg[r][ht][e + 3];
                }
            }
            float part = (p0 + p1) + (p2 + p3);
            part += __shfl_xor(part, 32);
            if (r < nrows && lane < 32 && n >= ks_r[r] && n < ke_r[r])
                out[(size_t)(m_base + r) * out_stride + n] = part;
        }
    };

    // UNROLL-wide, 2-deep software pipeline, warp-strided tiles.
    const int it0 = tile_start + warp;
    int4v Kc0[UNROLL], Kc1[UNROLL]; int Sc0[UNROLL], Sc1[UNROLL];
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
        const int t = it0 + u * WARPS;
        if (t < tile_end) gather(t, Kc0[u], Kc1[u], Sc0[u], Sc1[u]);
    }
    for (int it = it0; it < tile_end; it += UNROLL * WARPS) {
        int4v Kn0[UNROLL], Kn1[UNROLL]; int Sn0[UNROLL], Sn1[UNROLL];
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const int t = it + (UNROLL + u) * WARPS;
            if (t < tile_end) gather(t, Kn0[u], Kn1[u], Sn0[u], Sn1[u]);
        }
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const int t = it + u * WARPS;
            if (t < tile_end) compute(t, Kc0[u], Kc1[u], Sc0[u], Sc1[u]);
        }
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            Kc0[u] = Kn0[u]; Kc1[u] = Kn1[u]; Sc0[u] = Sn0[u]; Sc1[u] = Sn1[u];
        }
    }
}

#ifndef FP4_DENSE_MTILE
#define FP4_DENSE_MTILE 2   // m-rows fused per block in the dense32 path
#endif

extern "C" void launch_fp4_mqa_dense32(
    const uint8_t* q_p, const uint8_t* q_s,
    const uint8_t* kv_p, const uint8_t* kv_s,
    const float* weights, const int* cu_starts, const int* cu_ends,
    float* out, int M, int Nkv, int H,
    int kv_p_stride, int kv_s_stride, int out_stride, int split_kv, long stream)
{
    hipStream_t s = reinterpret_cast<hipStream_t>(stream);
    dim3 block(WARPS * LANES);

    // M-tiled path: amortize the (M-redundant) KV stream across MROWS rows.
    // Only beneficial when the launch already saturates the machine (split_kv
    // small) and there are enough rows to tile; otherwise fall back to 1 row.
#if FP4_DENSE_MTILE > 1
    if (split_kv == 1 && M >= FP4_DENSE_MTILE * 2) {
        const int MR = FP4_DENSE_MTILE;
        const int num_mb = (M + MR - 1) / MR;
        dim3 grid(num_mb * split_kv);
        auto launch_mt = [&](auto kern) {
            kern<<<grid, block, 0, s>>>(
                q_p, q_s, kv_p, kv_s, weights, cu_starts, cu_ends, out,
                M, Nkv, kv_p_stride, kv_s_stride, out_stride, split_kv);
        };
        if (H == 64)       { launch_mt(fp4_mqa_dense32_mtile_kernel<64, FP4_DENSE_MTILE>);  return; }
        else if (H == 32)  { launch_mt(fp4_mqa_dense32_mtile_kernel<32, FP4_DENSE_MTILE>);  return; }
        else if (H == 128) { launch_mt(fp4_mqa_dense32_mtile_kernel<128, FP4_DENSE_MTILE>); return; }
    }
#endif

    dim3 grid(M * split_kv);
    auto launch = [&](auto kern) {
        kern<<<grid, block, 0, s>>>(
            q_p, q_s, kv_p, kv_s, weights, cu_starts, cu_ends, out,
            M, Nkv, kv_p_stride, kv_s_stride, out_stride, split_kv);
    };
    if (H == 64)       launch(fp4_mqa_dense32_kernel<64>);
    else if (H == 32)  launch(fp4_mqa_dense32_kernel<32>);
    else if (H == 128) launch(fp4_mqa_dense32_kernel<128>);
}

// ==========================================================================
// Dense MXFP4 MQA-logits, 32x32x64 core + async-copy LDS staging (CDNA4)
// --------------------------------------------------------------------------
// EXPERIMENTAL / NOT THE DEFAULT (core="32" is). gluon-style async-copy: each
// warp async-copies its next K tile global->LDS (global_load_lds, a vmcnt-
// tracked side effect off the register pipeline) and computes the current tile
// from LDS, with an NB-deep pipeline keeping NB-1 copies in flight so memory can
// run during MFMA/VALU.
//
// RESULT: it does NOT beat the register-pipelined core="32" -- measured ~0.96-
// 0.99x (≈4% slower) across all sizes incl. 2048x8192, and pipeline depth
// (NB=2..6) does not move bandwidth. BW stays pinned at ~8 TB/s == core="32".
// Conclusion: the kernel is not memory-*latency* bound, so decoupling load issue
// from compute buys nothing. The global-K read ceiling *under compute* is
// ~8.25 TB/s at 4-wave occupancy, and routing K through LDS only adds traffic
// (global read + LDS write + LDS read) on top of that same ceiling (the
// "~13 TB/s load-only" probe was compute-free and not representative). Raising
// the ceiling needs higher memory-level parallelism (more occupancy), which the
// 128-VGPR register pipeline rules out. Kept as a reference for the async-copy
// machinery; safe to delete.
//
// NOTE: global_load_lds writes LDS *contiguously* (lane i -> base + i*size); the
// per-lane dst offset is ignored (verified), so a padded/swizzled layout is not
// possible via the copy. We store 64B/token contiguous + a separate 4B/token
// scale region; the per-lane *global src* may gather, used to clamp OOB tokens.
// vmcnt-only wait (gfx9/CDNA s_waitcnt encoding): bits[3:0]+[15:14]=VMCNT,
// [6:4]=EXPCNT(max=7), [13:8]=LGKMCNT(max=0x3f). 0x3f70 ignores exp/lgkm.
// s_waitcnt needs a compile-time immediate, so N is a template constant.
template <int N>
__device__ __forceinline__ void wait_vmcnt() {
    __builtin_amdgcn_s_waitcnt(0x3f70 | (N & 0xf) | (((N >> 4) & 0x3) << 14));
}

template <int HEADS, int NB>
__global__ void fp4_mqa_dense32_lds_kernel(
    const uint8_t* __restrict__ q_p,        // [M, H, D/2]
    const uint8_t* __restrict__ q_s,        // [M, H, D/32]
    const uint8_t* __restrict__ kv_p,       // [Nkv, D/2]
    const uint8_t* __restrict__ kv_s,       // [Nkv, D/32]
    const float*   __restrict__ weights,    // [M, H]
    const int*     __restrict__ cu_starts,  // [M]
    const int*     __restrict__ cu_ends,    // [M]
    float*         __restrict__ out,        // [M, Nkv]  (pre-filled -inf)
    int M, int Nkv, int kv_p_stride, int kv_s_stride, int out_stride, int split_kv)
{
    constexpr int DP = 64, DS = 4, HT2 = HEADS / 32;
    constexpr int BN = 32;                 // tokens per tile
    constexpr int DV = DP / 16;            // int4v chunks per token data (=4)
    constexpr int LPT = 3;                  // global_load_lds per tile (2 data + 1 scale)
    __shared__ int4v kdat[WARPS * NB * BN * DV];   // 64B/token contiguous
    __shared__ int   kscl[WARPS * NB * BN];        // 4B/token scales

    const int pid   = blockIdx.x;
    const int m     = pid % M;
    const int split = pid / M;

    const int ks = cu_starts[m];
    const int ke = cu_ends[m];
    if (ks >= ke) return;

    const int t_lo = ks >> 5;
    const int t_hi = (ke + 31) >> 5;
    const int total_tiles = t_hi - t_lo;
    const int tiles_per_split = (total_tiles + split_kv - 1) / split_kv;
    const int tile_start = t_lo + split * tiles_per_split;
    const int tile_end   = min(tile_start + tiles_per_split, t_hi);
    if (tile_start >= tile_end) return;

    const int warp = threadIdx.x / LANES;
    const int lane = threadIdx.x % LANES;
    const int rc   = lane & 31;
    const int hi   = lane >> 5;

    // Q + scales + weights for row m (once, reused). Same layout as core="32".
    const uint8_t* q_p_base = q_p + (size_t)m * HEADS * DP;
    const uint8_t* q_s_base = q_s + (size_t)m * HEADS * DS;
    const float*   w_base   = weights + (size_t)m * HEADS;
    int4v qreg[HT2][2];
    int   qscale[HT2][2];
    float wreg[HT2][16];
#pragma unroll
    for (int ht = 0; ht < HT2; ++ht) {
        const int head = 32 * ht + rc;
        const uint8_t* qb = q_p_base + (size_t)head * DP + 32 * hi;
        qreg[ht][0]   = load16v4(qb);
        qreg[ht][1]   = load16v4(qb + 16);
        qscale[ht][0] = (int)q_s_base[(size_t)head * DS + 2 * hi];
        qscale[ht][1] = (int)q_s_base[(size_t)head * DS + 2 * hi + 1];
#pragma unroll
        for (int e = 0; e < 16; ++e)
            wreg[ht][e] = w_base[32 * ht + 8 * (e >> 2) + 4 * hi + (e & 3)];
    }
    float* out_row = out + (size_t)m * out_stride;

    const int dbase = (warp * NB) * (BN * DV);   // int4v index, +buf*(BN*DV)
    const int sbase = (warp * NB) * BN;          // int   index, +buf*BN

    // Async-copy one 32-token K tile (data + scales) into LDS buffer `buf`.
    // dst is contiguous by lane; src is a per-lane gather (clamped at OOB).
    auto stage = [&](int t_idx, int buf) {
        const int tilebase = t_idx * BN;
        const int db = dbase + buf * (BN * DV);
#pragma unroll
        for (int c = 0; c < 2; ++c) {                 // 2 x (64 lanes x 16B) = 2048B
            const int ci  = c * 64 + lane;             // 16B chunk index in tile
            const int tok = ci >> 2;                   // /DV
            const int r   = ci & 3;
            const int n   = tilebase + tok;
            const uint8_t* src = kv_p + (size_t)(n < Nkv ? n : 0) * kv_p_stride + r * 16;
            __builtin_amdgcn_global_load_lds(
                (const int*)src, (int*)&kdat[db + ci], 16, 0, 0);
        }
        if (lane < BN) {                               // 32 x 4B scales
            const int n = tilebase + lane;
            const uint8_t* src = kv_s + (size_t)(n < Nkv ? n : 0) * kv_s_stride;
            __builtin_amdgcn_global_load_lds(
                (const int*)src, (int*)&kscl[sbase + buf * BN + lane], 4, 0, 0);
        }
    };

    auto compute = [&](int t_idx, int buf) {
        const int db = dbase + buf * (BN * DV) + rc * DV;
        const int4v b0 = kdat[db + 2 * hi];
        const int4v b1 = kdat[db + 2 * hi + 1];
        const int   sc = kscl[sbase + buf * BN + rc];
        const int   s0 = (sc >> (16 * hi)) & 0xFF;
        const int   s1 = (sc >> (16 * hi + 8)) & 0xFF;
        const int8v K0 = widen(b0), K1 = widen(b1);
        float p0 = 0.f, p1 = 0.f, p2 = 0.f, p3 = 0.f;
#pragma unroll
        for (int ht = 0; ht < HT2; ++ht) {
            float16v acc;
#pragma unroll
            for (int e = 0; e < 16; ++e) acc[e] = 0.f;
            acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
                widen(qreg[ht][0]), K0, acc, 4, 4, 0, qscale[ht][0], 0, s0);
            acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
                widen(qreg[ht][1]), K1, acc, 4, 4, 0, qscale[ht][1], 0, s1);
#pragma unroll
            for (int e = 0; e < 16; e += 4) {
                p0 += fmaxf(acc[e + 0], 0.0f) * wreg[ht][e + 0];
                p1 += fmaxf(acc[e + 1], 0.0f) * wreg[ht][e + 1];
                p2 += fmaxf(acc[e + 2], 0.0f) * wreg[ht][e + 2];
                p3 += fmaxf(acc[e + 3], 0.0f) * wreg[ht][e + 3];
            }
        }
        float part = (p0 + p1) + (p2 + p3);
        part += __shfl_xor(part, 32);
        const int n = t_idx * BN + rc;
        if (lane < 32 && n >= ks && n < ke)
            out_row[n] = part;
    };

    // NB-deep async pipeline (this warp owns tiles tile_start+warp, +WARPS, ...).
    // Keep NB-1 tile copies in flight so memory runs continuously while compute
    // lags behind in LDS. Steady-state wait is a constant vmcnt(LPT*(NB-1));
    // the last NB-1 tiles are drained with s_waitcnt(0).
    const int tb = tile_start + warp;
    const int nt = (tile_end - tb + WARPS - 1) / WARPS;   // tiles for this warp
    if (nt <= 0) return;
    const int pr = min(NB - 1, nt);
#pragma unroll 1
    for (int p = 0; p < pr; ++p) stage(tb + p * WARPS, p % NB);
    int k = 0;
#pragma unroll 1
    for (; k + (NB - 1) < nt; ++k) {
        stage(tb + (k + NB - 1) * WARPS, (k + NB - 1) % NB);
        wait_vmcnt<LPT * (NB - 1)>();
        compute(tb + k * WARPS, k % NB);
    }
    __builtin_amdgcn_s_waitcnt(0);
#pragma unroll 1
    for (; k < nt; ++k) compute(tb + k * WARPS, k % NB);
}

extern "C" void launch_fp4_mqa_dense32_lds_nb(
    const uint8_t* q_p, const uint8_t* q_s,
    const uint8_t* kv_p, const uint8_t* kv_s,
    const float* weights, const int* cu_starts, const int* cu_ends,
    float* out, int M, int Nkv, int H,
    int kv_p_stride, int kv_s_stride, int out_stride, int split_kv, int nbuf, long stream)
{
    dim3 grid(M * split_kv);
    dim3 block(WARPS * LANES);
    hipStream_t s = reinterpret_cast<hipStream_t>(stream);
    auto launch = [&](auto kern) {
        kern<<<grid, block, 0, s>>>(
            q_p, q_s, kv_p, kv_s, weights, cu_starts, cu_ends, out,
            M, Nkv, kv_p_stride, kv_s_stride, out_stride, split_kv);
    };
#define LDS_DISPATCH(NB) \
    if (H == 64)       launch(fp4_mqa_dense32_lds_kernel<64,  NB>); \
    else if (H == 32)  launch(fp4_mqa_dense32_lds_kernel<32,  NB>); \
    else if (H == 128) launch(fp4_mqa_dense32_lds_kernel<128, NB>);
    switch (nbuf) {
        case 2:  { LDS_DISPATCH(2);  break; }
        case 3:  { LDS_DISPATCH(3);  break; }
        case 6:  { LDS_DISPATCH(6);  break; }
        case 8:  { LDS_DISPATCH(8);  break; }
        default: { LDS_DISPATCH(4);  break; }
    }
#undef LDS_DISPATCH
}

extern "C" void launch_fp4_mqa_dense32_lds(
    const uint8_t* q_p, const uint8_t* q_s,
    const uint8_t* kv_p, const uint8_t* kv_s,
    const float* weights, const int* cu_starts, const int* cu_ends,
    float* out, int M, int Nkv, int H,
    int kv_p_stride, int kv_s_stride, int out_stride, int split_kv, long stream)
{
    launch_fp4_mqa_dense32_lds_nb(q_p, q_s, kv_p, kv_s, weights, cu_starts, cu_ends,
        out, M, Nkv, H, kv_p_stride, kv_s_stride, out_stride, split_kv, 4, stream);
}
