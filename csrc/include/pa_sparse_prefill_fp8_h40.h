// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// FP8 sparse paged prefill attention for DeepSeek-V4 on gfx950 (device side).
//
// The accumulator is never rescaled.  Online softmax normally multiplies all
// 256 f32 of v_o by exp2(m_old - m_new) once per tile; because v_o is bigger
// than the ArchVGPR file it lives in AGPRs, so each of those multiplies drags a
// v_accvgpr_read and a v_accvgpr_write with it.  That was 40% of all VALU.
// Instead v_o is pinned to a fixed log2 frame m_ref and the per-tile correction
// -- made an exact power of two by flooring the running max -- rides the PV
// mfma's scale_b field, which the hardware applies for free.  See the softmax
// phase-1 block for why scale_b can carry a *per-head* value and scale_a cannot.
// Measured, N=1024/4096 at H=128: 250/856 -> 220/768 us, VALU 3472 -> 2994
// instructions, v_accvgpr 927 -> 549, v_mul_f32 224 -> 28, s_nop 271 -> 69.
//
// Bound: 127 + (m_final - m_ref) must fit an E8M0 byte and v_o must stay in f32
// range, so the running max may grow at most ~96 octaves above the first tile's.
// That is a softmax whose late logits beat its early ones by e^66; real
// attention is nowhere near it, but the failure mode would be silent.
//
// The alternative -- deriving m_ref in the prologue from a static upper bound
// (|S| <= 448 terms x 448.2^(qe-127) x 448.2^(max_e-127), so
// m_ref = 27 + qe + max_e - 254 + frexp_exp(c_row)) -- was implemented and is
// bit-identical in accuracy.  It makes dexp <= 0 by construction, drops m_row
// and the first-tile conditional, and lets the sink be folded into l_row before
// the loop.  It is also 1% slower at N=4096 (748.6 vs 755.6 us, interleaved
// 5-rep means, 4 of 5 pairs) and spills 12 B/lane, and the failure mode it
// removes is no more reachable than the one it introduces (the bound
// undershooting by 2^126), so it is not used.
//
// Fusing the exp2 into the QK loop -- which that static frame makes possible,
// since bias then stops depending on the tile's row max -- is much worse: 286 vs
// 218 us.  Reading acc[] right after its own mfma chain exposes the matrix-pipe
// latency once per n-subtile, and that costs far more than the ~2% of VALU it
// moves into the shadow.
//
// Cross-tile software pipelining (PV lagging one tile behind QK, so the softmax
// VALU overlaps the previous tile's PV mfmas) needs three KV buffers, because
// PV(i), QK(i+1) and the copy for i+2 would all be live at once.  It does not
// fit and cannot be made to fit: three buffers of 128 tokens x 448 B is 172,032
// bytes with zero padding and no rope tile at all, against a 163,840 byte LDS.
// Storing the tile as fp6 instead (0.75 B/element, and gfx950 does have
// ds_read_b96_tr_b6) would fit at 145,408 bytes -- but the prize is not there.
// Measured directly, without building any of it: feeding the PV the *previous*
// tile's P, which breaks exactly the dependency a third buffer would break, is
// worth 2.0% (795.6 -> 779.8 us at N=4096, same knobs), and it needs 16 more
// registers than the file has.  Worse, SQ_VALU_MFMA_COEXEC_CYCLES *falls* when
// the dependency is broken -- 11.6M -> 4.0M against 159.4M of mfma-busy -- so
// VALU/MFMA co-execution is limited by issue rules, not by the dependency, and
// the 27% that perfect overlap would be worth is not actually reachable.
//
// Moving the QK to v_mfma_scale_f32_32x32x64_f8f6f4 would make the contraction
// 7 exact k-steps of 64 instead of 4 padded slices of 128, removing the 448->512
// waste.  The instruction runs at the same rate (4759 vs 4728 TFLOPS measured),
// but the prize is small: ablating a *whole* d-slice -- twice the padding waste
// -- is worth only 1.4% / 1.1%, so the real gain is ~0.6%, against redesigning
// the LDS token permutation, the softmax reductions, the masking and the
// epilogue, since the 32x32 C layout does not match the 16x16x128 PV B layout
// the way the 16x16 one does.
//
// Measured shares of wave-cycles after that change (rocprofv3, this kernel
// only): MFMA pipe busy 55.0%, ACTIVE_INST_VALU 15.7%, ACTIVE_INST_LDS 2.5%,
// WAIT_INST_LDS 0.3%, WAIT_ANY 47.6%.  So it is now roughly half MFMA-limited;
// the remaining headroom is in the QK -> softmax -> PV serial chain, not in any
// one instruction class.
//
// v_o = heads_per_wave * cols_per_wave / 64 = 256 for *any* 4-wave split of 128
// heads x 512 columns, and every decomposition that shrinks it is worse: Q_SUB=1
// with NUM_WARPS=4 doubles the KV traffic (391 vs 291 us at the old baseline),
// and NUM_WARPS=8 (2 waves/SIMD, i.e. AMD's ping-pong shape) halves the register
// budget to 256 -- it spills 64-96 B/lane *and* doubles LDS operand traffic
// because a wave then has only one Q sub-tile to amortise each K/V read over:
// 270 vs 220 us, lds_active 620M -> 1237M.  Hoisting the QK d-slice loop out of
// the n-subtile loop was a wash, and 4-way QK ILP is 2% slower than 2-way.
//
// Cost breakdown, measured by ablation at N=4096 / H=128 / top-k 1024 against a
// 1123 us baseline (each ablation is timing-only, the results are wrong):
//   KV async copy      -96 us   of which only 20 is the top-of-tile vmcnt wait;
//                               the rest is LDS write-port contention
//   RoPE async copy    -43
//   PV-RoPE (all)      -80      reads 24, the 32 bf16 mfmas and their bf16 P 56
//   QK NoPE reads      -53
//   QK RoPE reads      -16
//   PV NoPE tr_b8      -0       fully hidden behind its own mfmas
//   staging barrier    -3
// (these were taken before the accumulator rescale was removed; the shares move
// but the ordering does not)
// Pure mfma time is ~260 us, and peak LDS is 192 B/clk/CU (b128) / 163 (tr_b8),
// against ~36 B/clk/CU here -- so this is not an LDS bandwidth wall.  Ablating
// the softmax transcendentals is not informative: removing them perturbs
// register allocation enough to make the kernel 40% *slower*.
//
// Both GEMMs run on v_mfma_scale_f32_16x16x128_f8f6f4 — measured at 4.73
// PFLOPS on MI355X vs 2.28 PFLOPS for v_mfma_f32_16x16x32_bf16.  The legacy
// v_mfma_f32_16x16x32_fp8_fp8 encoding runs at the *bf16* rate on gfx950
// (measured 2.30 PFLOPS), so it is deliberately not used.
//
// Because the PV contraction dimension is the token axis, K=128 forces
// KV_TILE=128.  Everything else follows from that.
//
// Wave decomposition
// ------------------
// 4 waves of 64, each owning Q_SUB = 2 mfma tiles of 16 heads, so a block still
// covers 128 heads and the KV traffic per block is unchanged.  Two consequences:
//
//   * 4 waves on 4 SIMDs is one wave per SIMD, which raises the VGPR budget from
//     256 to 512.  The 8-wave version needed 336 registers of persistent state
//     alone once RoPE was added and spilled 186 of them, costing 2.4x.
//   * the K and V operands are loaded from LDS once and feed both mfma tiles, so
//     the LDS read traffic per output element halves.
//
// The cost is no second wave to hide latency on a SIMD; that is paid back by the
// two independent mfma chains inside each wave.
//
// DeepSeek-V4 H40 input format (matches aiter's pa_sparse_prefill_fp8_opus):
//   q_nope           : [N, H, 448] bf16 -- packed in the prologue, one E8M0
//                      per head; no separate pack pass exists
//   kv_nope          : [.., 512] fp8 rows packed as
//                      448 NoPE fp8 | 14 E8M0 block scales (block = 32) | 50 pad
//   q_rope / kv_rope : [.., 64] bf16
//
// Scale routing
// -------------
//   Q NoPE : per-32 E8M0 -> hardware scale_b, zero VALU.  Needs the verified
//            placement d = 128*ds + 64*(j/16) + 16*g + (j%16), where lane group
//            g supplies the byte of block 4*ds+g.
//   K NoPE : requantised to ONE per-token E8M0 at LDS staging (an exact
//            power-of-two exponent shift), then fed as hardware scale_a.
//   V NoPE : same per-token exponent, folded into P (which is why K/V had to be
//            brought to per-token: the PV contraction is the token axis, so a
//            scale that varies along d cannot be absorbed by the MFMA).
//
// QK: S[i,t] = (q_scale_i * kv_scale_t * softmax_scale) * <q8_i, k8_t>.
//     kv_scale_t varies along t so it is a per-element multiply on S;
//     q_scale_i is constant along t and folds into the exp2 argument as one
//     FMA per element.
//
// PV: O[i,:] = sum_t P[i,t] * kv_scale_t * v8[t,:].  The per-token factor is
//     folded into P before quantising it to fp8:
//
//         P8[i,t] = fp8( P[i,t] * kv_scale_t * alpha )
//         O[i,:]  = (1/alpha) * sum_t P8[i,t] * v8[t,:]
//
//     alpha is one kernel-wide power of two, chosen host-side as
//     2^floor(log2(240 / max kv_scale)).  Since P <= 1 exactly, the product is
//     bounded by 240 < 448, so no clamping is required.  The softmax
//     denominator accumulates the *unscaled* P and the final normalisation
//     divides by (alpha * l_row), putting alpha back.
#pragma once

#include <opus/opus.hpp>
#include <bit>
#include <type_traits>

// The whole kernel lives in its own namespace: aiter's pa_sparse_prefill_opus.h
// already defines a `pa_fp8_kargs` in the same translation unit, and its IMPL
// section puts bf16_t / fp8_t at global scope.
namespace pa_fp8_h40 {

// Minimum head count for which this kernel is used -- the throughput-optimal
// boundary, measured.  The block is 128 heads wide, so H below this wastes it.
//
// 64, from the 36-point scenario matrix (2026-07-30, MI355X idle,
// dsv4-fp8/scenarios.py): this kernel wins 1.15-1.27x for H >= 64 and N >= 256
// across every length / topk / context / extend, and *loses* below H=64, where
// aiter's T_M=1 fp8 kernel is the better route.
//
// An earlier, narrower probe (N=4096 only, against bf16: H=32 0.893x a loss,
// H=40 1.114x) put the boundary in (32, 40] and this was 40.  The scenario
// matrix supersedes it -- it sweeps N as well, and N is the second dimension
// that moves the verdict.  Keep in sync with PA_FP8_H40_MIN_H in
// aiter/ops/pa_sparse_prefill_opus.py.
#ifndef PA_FP8_H40_MIN_H
#define PA_FP8_H40_MIN_H 16
#endif

// Prescaled KV (one E8M0 per token) is the op's input contract, so the
// in-kernel requant arm is never built.

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
    // Per segment: sglang's SWA and sparse pools are the same 576-byte grid
    // with the same per-64 UE8M0 page-tail scales (proved by
    // triton_fused_store_flashmla's constants) and differ only in page_size,
    // so the row/rope strides stay shared and only these three split.
    int   sgl_page_shift[2];      // [0] prefix / unified, [1] extend
    int   sgl_rows_per_page[2];
    int   sgl_scale_off[2];
    // Dense-index mode.  sglang hands out [N, topk] indices plus a per-query
    // length, not a CSR indptr; compacting that costs an O(N*topk) copy per
    // layer (16 MB at N=4096, topk=1024, so ~1 GB over 61 layers).  When
    // kv_lens_* is non-null the segment reads indices at q*kv_stride_q_* for
    // kv_lens_*[q] entries and the indptr is ignored.
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
    // Stage the epilogue's output through LDS so the global stores coalesce.
    //
    // The mfma C layout gives lane (c,g) four consecutive columns of head c, so
    // a direct store scatters 64 lanes over 16 separate 32 B segments 1024 B
    // apart.  Measured on the dev harness 2026-08-04 by shrinking the row
    // stride to 16 B -- same instructions, same bytes, same immediates, only
    // the address pattern -- that divergence is worth 31 us at N=1024, i.e.
    // 78% of the entire epilogue.  Neither instruction count nor bytes explain
    // any of it: batching the converts into 16 independent chains, and halving
    // the stored bytes, each moved it under 1 us.
    //
    // Storing coalesced *directly* is what the register file refuses -- the row
    // offsets pass the 12-bit immediate, so every store needs its own address,
    // and both attempts spilled exactly 1032 B/lane and ran 1.8x slower.  By
    // the epilogue all of LDS is dead, so transpose through it instead: each
    // wave writes its 16 head rows in the layout it already has, then reads
    // back 16 B/lane so 64 lanes cover one whole 1024 B row per store.  The
    // store phase holds two registers, not 64, which is why it does not spill.
#ifndef PA_EPI_LDS
#define PA_EPI_LDS 1
#endif
    // bf16 per staged row.  Swept 512/516/520/528/544 at the c128 shape:
    // 187.9 / 169.5 / 167.1 / 170.3 / 175.0 us.  The unpadded 512 is *worse
    // than not doing this at all* -- a 1024 B pitch puts all 16 c-lanes on the
    // same LDS bank.
#ifndef PA_EPI_PITCH
#define PA_EPI_PITCH 520
#endif
#ifndef PA_NO_COLLAPSE
// Run with the KV cache left exactly as written: collapse the staged tile in
// LDS (PA_LDS_REQUANT) and stop depending on a global max exponent.
//
// max_e existed only to put P in e4m3's range, and it was produced by the same
// cache-rewriting pass we are removing.  A *per-tile* max does the job -- P is
// built from that tile's own exponents -- and the frame difference between
// tiles rides the PV mfma's scale_b, which already carries `127 + dexp`:
//
//   accumulated = p * v8 * 2^(e_tok - max_e_t + 7 - dexp + s)
//   want        = true * 2^(134 - MAXE),  true = p * v8 * 2^(e_tok - 127)
//   =>  s = max_e_t + dexp - MAXE,   scale_b byte = 127 + s
//
// MAXE is free; 127 makes the byte `max_e_t + dexp` and the epilogue's
// inv_alpha the constant 2^-7.
#define PA_NO_COLLAPSE 1
#endif
#if PA_NO_COLLAPSE
#undef PA_LDS_REQUANT
#define PA_LDS_REQUANT 1
#endif
#ifndef PA_LDS_REQUANT
// Collapse the tile to one exponent per token *in LDS*, so the KV cache itself
// is never rewritten and vLLM needs no hook at the write sites.  O(N*topk)
// rather than O(rows): a row gathered by a hundred query tokens is requantised
// a hundred times, which is what the +52.7% buys.  On a cache that some other
// pass already flattened every shift is zero and this is a no-op, so measuring
// it needs a genuinely unflattened cache -- see op_tests.
#define PA_LDS_REQUANT 0
#endif
// Cache policy for the workgroup's *streaming* traffic (aux bit 1 = NT on
// gfx950: the line still allocates but is marked for eviction first).
//
// Measured under the production index distribution (op_tests use uniform random
// indices, where adjacent query tokens share ~0 rows and NOTHING about cache
// reuse can be observed -- see the note on --real in paged_bench.py):
//
//                      time      TCC_MISS   TCP_TCC_READ_REQ  TCP_PENDING_STALL
//   off              2.6906 ms       --            --                --
//   Q + out          2.6737  -0.63%  -15.6%      +12.8%             +1.4%
//   out only         2.6609  -1.10%   -0.3%       -0.0%             -2.6%   <- default
//
// NT on a *load* does cut L2 misses, but it also makes the line evict-first in
// L1, so the lanes sharing a 128 B line stop coalescing there and the request
// count to L2 goes up by more than the misses go down.  A store has nothing to
// reuse in L1, so NT there is free: misses and requests are unchanged and the
// whole gain is TCP no longer holding the line.  Hence Q=0, out=1.
// The PV batch waits per-tr_load instead of draining the whole batch: the j-th
// mfma only needs vv[j], and LDS retires in order.  Measured -0.58% (2.6594 ->
// 2.6439 ms, real indices, 4 reps interleaved), output bit-identical.
//
// This is only correct while the LGKM queue holds nothing but this batch's
// tr_loads.  SMEM retires OUT of order, so one s_load between two tr_loads
// makes the partial count return early and the mfma reads stale V -- corrupting
// the NoPE columns only (P and the RoPE PV stay bit-exact), which is why it
// looks like a codegen bug.  Three unrelated scheduling changes tripped it
// before, and `_tr_load` is inline asm the compiler cannot see, so nothing in
// the source prevents a recurrence.
//
// Hence check_pv_wait.py, run from sweep.sh on every build: it scans the ISA
// for exactly this condition and fails the build.  Do NOT enable this without
// that guard in the build path.
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
// Keep the `k == 0 -> nothing to do` skip.  Off by default, for two measured
// reasons that both cut against the obvious reading:
//
//   * It is a *divergent* branch.  k is per-lane and a wave's 64 lanes are 64
//     different tokens, so the exec mask only masks lanes off -- the
//     instructions issue regardless.  It buys nothing unless every lane in the
//     wave has k == 0 at once, which happens only on a cache some other stage
//     already flattened, i.e. never under PA_NO_COLLAPSE.  Dropping it is 3.5%.
//   * It is not numerically neutral, and not in the direction one would guess.
//     shift_exp_dword(dw, 0) is NOT the identity: it flushes the fifteen e4m3
//     subnormal byte values to zero.  The skip therefore *preserves* subnormals
//     in the one block per token that already carries the token max, and every
//     other block loses them; dropping it makes that uniform.  Measured against
//     the bit-identical arm: 99.0% of rows unchanged, max rel L2 7.4e-04 on the
//     rest -- 1/34 of the kernel's own 2.5e-02 fp8 floor.
//
// Set to 1 for output bit-identical to the kernel this came from.
#ifndef PA_RQ_BRANCH
#define PA_RQ_BRANCH 0
#endif
// ATOMs whose LDS read is issued before the drain.  With RQ_BATCH 1 the read
// and the write share one register quad, so the pass is a chain of
// O_TILES/RQ_SPLIT full round trips; a batch buys 4 VGPRs per extra ATOM and
// pays one drain for all of them.  Must divide O_TILES/RQ_SPLIT.
// Fold the tile-frame maximum in registers before it reaches LDS.  128 threads
// issuing atomicMax at one LDS address serialise in hardware, and that turned
// out to be the single largest item in the no-collapse overhead: folding across
// g (two permlanes) is 8.7%, and folding the row as well (four DPP row_shr,
// leaving one atomic per wave instead of 128 per tile) another 2.5%.  All of it
// is VALU on values already in registers -- no LDS, no extra live range, which
// is what makes it affordable on a kernel with no register headroom.
//
// Exact: max over the tile is max over the per-wave maxima, and invalid slots
// contribute 0.  Verified bitwise against the unfolded arm.
#ifndef PA_RQ_FOLD
#define PA_RQ_FOLD 1
#endif
#ifndef PA_RQ_BATCH
#define PA_RQ_BATCH 1
#endif
#ifndef PA_PV_BATCH
#define PA_PV_BATCH 4   // swept 1/2/4/7/14 -> 322/309/311/312/388 us at N=1024
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
#ifndef PA_PAD
#define PA_PAD 64  // swept 0..192 with conflict counters: timing flat, 64 marginally best
#endif
    // Padding was swept 0..192 and makes no measurable difference now that the
    // QK skips its tail slice; 0 buys 2 KB of LDS.
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
    static constexpr int SCALE_BYTES = EXP_BYTES;
    // The rope tile can only be double-buffered because the kv_scale gather is
    // gone; the per-32 scale array it needed pushed LDS over 160 KB.
#ifndef PA_ROPE_BUFS
// Double-buffering the rope tile is within noise (<1%) and puts LDS at exactly
// 163840, so it is off by default; set PA_ROPE_BUFS=2 to try it.
#define PA_ROPE_BUFS 1
#endif
    static constexpr int ROPE_BUFS = PA_ROPE_BUFS;
    // K operands kept in flight across n-subtiles.  Only affordable when the
    // kv_scale gather is gone; with e_pf still live the extra 32 registers cost
    // 27 spills and 7%.
// On by default in aiter: the flat [rows, 512] packing is the *degenerate*
// parameterisation of the paged one (page_shift 0, rows_per_page 1,
// scale_off 448 reproduce grid_row(g)=g and exp_off(g)=g*512+448 exactly), so
// one binary serves both sglang's DSv4 pool and aiter's own layout.  The dev
// harness still builds with 0 to measure what the exponent gather costs.
// Which KV row layout the kernel is bound to.  0 = flat 512 B rows, where the
// per-token E8M0 rides the tile itself; 1 = sglang's 576 B page grid, where
// bytes 448..511 are RoPE bf16 and the exponent has to be gathered from the
// page's own scale region on a two-tile-ahead pipeline.
//
// That gather is not free: at H=128, T=1024 it costs a flat caller 1.081x at
// 160 rows/token, 1.111x at 528, 1.109x at 640 and 1.126x at 1152, with
// bit-identical output.  Default to flat, which is what every caller here is.
//
// Serving both layouts from one build means templating on this and dispatching
// at launch.  Two of the eight sites below are type-dependent (exp_pf_t is int
// under paged and i32x4 under flat), which is why they are `#if` today; do that
// work when there is a paged caller to test the other arm against.
#ifndef PA_QK_PIPE
// The 8 B/token staging is still tiny against LDS; keep the K pipeline on.
#define PA_QK_PIPE ((EXP_BYTES <= KV_TILE * 8) ? 2 : 1)
#endif
    static constexpr int QK_PIPE = PA_QK_PIPE;
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

__global__ void pa_prefill_fp8_kernel(pa_fp8_kargs kargs);



#if !defined(__HIP_DEVICE_COMPILE__) || !defined(__gfx950__)
__global__ void pa_prefill_fp8_kernel(pa_fp8_kargs) {}

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
    return __builtin_amdgcn_mfma_f32_16x16x32_bf16(a, b, c, 0, 0, 0);
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
    auto grid_row = [&](int gid) -> __SIZE_TYPE__ {
        if constexpr (!T::SGL_PAGED) return (__SIZE_TYPE__)gid;
        else return (__SIZE_TYPE__)(gid >> sgl_page_shift)
                        * (__SIZE_TYPE__)sgl_rows_per_page
                    + (__SIZE_TYPE__)(gid & sgl_mask);
    };
    // Byte offset of a token's E8M0 slot inside its page's scale region.  After
    // the collapse pass all seven per-64 exponents are equal, so byte 0 is the
    // token exponent.
    auto exp_off = [&](int gid) -> __SIZE_TYPE__ {
        return (__SIZE_TYPE__)(gid >> sgl_page_shift)
                   * (__SIZE_TYPE__)sgl_rows_per_page
                   * (__SIZE_TYPE__)kargs.stride_kv_row
             + (__SIZE_TYPE__)sgl_scale_off
             + (__SIZE_TYPE__)(gid & sgl_mask) * 8u;
    };

    // ---- per-lane LDS read bases ----------------------------------------
    // QK, verified placement: element j of lane (c,g) carries
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
    // kv_indices -> kv_scale -> async-copy is a chain of dependent global loads.
    // Left in place it costs ~1800 stalled cycles at the top of every tile and,
    // worse, delays the async copy that the whole pipeline hangs on.  So the
    // row indices run two tiles ahead in registers, and the gathered scale is
    // written to LDS only at the *end* of the tile, by which point its load has
    // long since landed.
    auto load_rows = [&](int tile_idx, int (&row)[T::SLOTS_PER_WAVE]) {
#pragma unroll
        for (int half = 0; half < T::SLOTS_PER_WAVE; ++half) {
            const int s_hi = T::SLOTS_PER_WAVE * warp + half;
            const int p    = 32 * (s_hi >> 2) + 4 * (s_hi & 3) + p_lane;
            row[half] = load(g_idx, tile_idx * T::KV_TILE + p)[0];
        }
    };
    auto issue_copy = [&](const int (&row)[T::SLOTS_PER_WAVE], int buf) {
        char* const kv_dst = smem + buf * T::TILE_BYTES;
#pragma unroll
        for (int half = 0; half < T::SLOTS_PER_WAVE; ++half) {
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
    // The 16 packed scale bytes of each token are gathered into their own LDS
    // array rather than read back out of the KV tile: the tile's tail gets
    // zeroed during requantisation, and an E8M0 byte of 127 is 0x7F, which is
    // NaN in e4m3 -- 0 * NaN would poison the MFMA even with Q's tail zeroed.
    const bool scale_lane = ((lane >> 3) == 0);
    using exp_pf_t = std::conditional_t<T::SGL_PAGED, i32x2, i32x4>;
    auto fetch_exps = [&](const int (&row)[T::SLOTS_PER_WAVE],
                          exp_pf_t (&e)[T::SLOTS_PER_WAVE]) {
#pragma unroll
        for (int half = 0; half < T::SLOTS_PER_WAVE; ++half) {
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
    auto commit_exps = [&](const exp_pf_t (&e)[T::SLOTS_PER_WAVE], int buf) {
        if (!scale_lane) return;
        char* dst = smem_sc0 + buf * T::SCALE_BYTES;
#pragma unroll
        for (int half = 0; half < T::SLOTS_PER_WAVE; ++half) {
            const int s_hi = T::SLOTS_PER_WAVE * warp + half;
            const int p = 32 * (s_hi >> 2) + 4 * (s_hi & 3) + p_lane;
#if PA_SGLANG_PAGED
            *reinterpret_cast<i32x2*>(dst + p * 8) = e[half];
#else
            *reinterpret_cast<i32x4*>(dst + p * 16) = e[half];
#endif
        }
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

    // Per-token exponent tables for buffer b.  All 14 E8M0 bytes already equal
    // the token exponent and the first sits at d = 448 = LDS cell 28, which
    // nothing zeroes now that the QK skips its tail read -- so it is already in
    // the tile and the separate kv_scale gather is pure overhead.
    //
    // The barrier after this pass costs 21 us (1.9%) at N=4096 and cannot be
    // merged away: staging reads all 128 tokens but a wave's s_waitcnt_vmcnt
    // only covers the slot blocks *it* copied, so every wave has to land its own
    // slice and then meet at a barrier.  Moving the pass to the previous tile's
    // tail to reuse the barrier there was tried: it is correct but 2% slower,
    // because the vmcnt that has to precede it then blocks the RoPE copy's
    // issue, and that copy loses the overlap it used to get.
#if PA_NO_COLLAPSE
    __shared__ int tile_max_e;
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
                            w[d] = pa_fp8::shift_exp_dword(w[d], k);
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
                                asm volatile("" : "+v"(t[j][d]) ::);
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
                                w[d] = pa_fp8::shift_exp_dword(t[j][d], k);
                        }
                    }
                }
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
                if ((lane & 15) == 15) atomicMax(&tile_max_e, etg);
#else
                if (tbase + tok < valid_kv_len) atomicMax(&tile_max_e, et);
#endif
#endif
                etok.template store<1>((u8_t)et, tok);
                etkt.template store<1>((u8_t)et, (tok & 15) * 8 + (tok >> 4));
            }
        }
    };

    int row_pf[T::SLOTS_PER_WAVE], row_pf2[T::SLOTS_PER_WAVE], rr_pf[T::RCOPY];
    exp_pf_t e_pf[T::SLOTS_PER_WAVE];
    load_rows(0, row_pf);
#if PA_SGLANG_PAGED
    fetch_exps(row_pf, e_pf);
    commit_exps(e_pf, 0);
#endif
    issue_copy(row_pf, 0);
    load_rope_rows(0, rr_pf);
    issue_rope_copy(rr_pf, 0);
    if constexpr (T::ROPE_BUFS == 2) load_rope_rows(1, rr_pf);
    load_rows(1, row_pf);   // indices for tile 1, awaited by the first barrier

    int buf = 0, rbuf = 0;
    for (int tile = 0; tile < num_tiles; ++tile) {
        char* const tilep = smem + buf * T::TILE_BYTES;
        auto s_kv   = make_smem(reinterpret_cast<fp8_t*>(tilep));
        auto s_exp  = make_smem(reinterpret_cast<u8_t*>(smem_sc0 + buf * T::SCALE_BYTES));
        auto s_etok = make_smem(reinterpret_cast<u8_t*>(tilep + T::ETOK_OFF));
        auto s_etkt = make_smem(reinterpret_cast<u8_t*>(tilep + T::ETKT_OFF));

        // One barrier per tile.  Reaching it means every wave has finished the
        // PV reads of the *other* buffer, so the next tile's copy can be issued
        // immediately after it — no second barrier needed.
        s_waitcnt_vmcnt(0_I);
        lds_barrier();
#if PA_NO_COLLAPSE
        // Reset before any atomicMax can land; one extra barrier, and a single
        // barrier either way is free here (measured).
        if (tid == 0) tile_max_e = 0;
        lds_barrier();
#endif

        const bool has_next = (tile + 1 < num_tiles);
#if PA_SGLANG_PAGED
        if (has_next) fetch_exps(row_pf, e_pf);
#endif
        // writes buf^1 while the staging pass below touches buf, so it can be
        // issued here and pick up the staging pass as extra cover
        if (has_next) issue_copy(row_pf, buf ^ 1);
        if constexpr (T::ROPE_BUFS == 2) {
            // double-buffered: no need to wait for every wave's PV-RoPE reads,
            // so the copy moves up here and gets a whole tile of cover
            if (has_next) issue_rope_copy(rr_pf, rbuf ^ 1);
        }

        // ---- staging requantisation: per-32 E8M0 -> one per-token exponent
        // REQ_SPLIT threads per token, thread q handling blocks q, q+2, ... and
        // zeroing tail cells 28+q, 30+q.  Each thread touches only its own
        // token, so no barrier is needed inside this pass.
        stage_exps(buf, tile * T::KV_TILE);
        lds_barrier();
#if PA_NO_COLLAPSE
        const int max_e_t = tile_max_e;
        const int pv_e0   = max_e_t;   // byte = 127 + (max_e_t + dexp - MAXE), MAXE = 127
#else
        const int max_e_t = max_e;
        const int pv_e0   = 127;
#endif

        if (has_next) {
            // The gather is issued above the requantisation pass and committed
            // below it, so its ~800 stalled cycles are covered and e_pf dies
            // before the QK instead of living across the whole tile -- 16
            // ArchVGPRs, which are the binding resource here.
#if PA_SGLANG_PAGED
            commit_exps(e_pf, buf ^ 1);       // buf^1 was last read by tile-1
#endif
            load_rows(tile + 2, row_pf2);     // OOB tiles read 0 from the buffer rsrc
            // double-buffered, the copy is issued at the *top* of the next
            // tile, so the indices must run one tile further ahead
            load_rope_rows(tile + (T::ROPE_BUFS == 2 ? 2 : 1), rr_pf);
        }

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
        auto load_k = [&](auto i_nt, i32x8 (&kk)[T::D_SLICES]) {
            constexpr int nt = decltype(i_nt)::value;
            constexpr int nt_off = (nt >> 1) * 4 * T::ROW + (nt & 1) * 64;
            static_for<T::D_SLICES>([&](auto i_ds) {
                constexpr int ds = i_ds.value;
                constexpr int off0 = nt_off + ds * 1024;
                // The hi half of the last slice is d = 448..511: the packed E8M0
                // bytes and the pad, never real data.  Q's matching half is
                // already zero, so feeding a zero register is identical -- and
                // skipping the load is 12.5% of the QK read traffic.  It is also
                // the only reader of cells 28..31, which is why the staging pass
                // no longer has to zero them.
                const i32x4 hi = (ds == T::D_SLICES - 1)
                    ? i32x4{0, 0, 0, 0}
                    : __builtin_bit_cast(i32x4,
                          s_kv.template _load<16>(qk_base + off0 + 4 * T::ATOM));
                kk[ds] = pack32(
                    __builtin_bit_cast(i32x4, s_kv.template _load<16>(qk_base + off0)), hi);
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

        i32x8  kkb[T::QK_PIPE][T::D_SLICES];
        bf16x8 krb[T::QK_PIPE][T::ROPE_KST];
        load_k(number<0>{}, kkb[0]);
        load_kr(number<0>{}, krb[0]);
        static_for<T::N_TILES>([&](auto i_nt) {
            constexpr int nt = i_nt.value;
            if constexpr (T::QK_PIPE == 2) {
                if constexpr (nt + 1 < T::N_TILES) {
                    load_k(number<nt + 1>{}, kkb[(nt + 1) & 1]);
                    load_kr(number<nt + 1>{}, krb[(nt + 1) & 1]);
                }
            } else if constexpr (nt > 0) {
                load_k(number<nt>{}, kkb[0]);
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
                    acc[qs] = mma_qk<ds>(kkb[(T::QK_PIPE == 2) ? (nt & 1) : 0][ds], v_q[qs][ds], acc[qs], sa, q_sb[qs]);
                });
            });
            // RoPE contributes to the very same S tile (same C layout)
            static_for<T::ROPE_KST>([&](auto i_st) {
                constexpr int st = i_st.value;
                static_for<T::Q_SUB>([&](auto i_qs) {
                    constexpr int qs = i_qs.value;
                    acc[qs] = mma_bf16(krb[(T::QK_PIPE == 2) ? (nt & 1) : 0][st], v_qr[qs][st], acc[qs]);
                });
            });
#pragma unroll
            for (int qs = 0; qs < T::Q_SUB; ++qs)
#pragma unroll
                for (int i = 0; i < 4; ++i) v_s[qs][nt * 4 + i] = acc[qs][i];
            // One n-subtile is 9 ds_reads (7 K slices + 2 RoPE) and 12 mfmas.
            // Left to itself the scheduler clusters the loads and then stalls on
            // s_waitcnt; describe the interleave explicitly.  0x008 = MFMA,
            // 0x100 = DS read.  Worth a consistent 0.8-1.4%.  Do not expect more
            // from scheduling here: SQ_WAIT_INST_LDS is only 2.0% of wave cycles,
            // i.e. LDS latency is already almost entirely hidden.
            static_for<PA_SCHED_REP>([&](auto) {
                __builtin_amdgcn_sched_group_barrier(0x100, PA_SCHED_DS, 0);
                __builtin_amdgcn_sched_group_barrier(0x008, PA_SCHED_MFMA, 0);
            });
        });

        // ---- per-token dequant, mask, online softmax ---------------------
        const bool last = (tile == num_tiles - 1);
        const int tile_base = tile * T::KV_TILE;
        i32x8 v_p[T::Q_SUB];
        float bias[T::Q_SUB], rsum[T::Q_SUB];
        int   dexp[T::Q_SUB];   // new_m - m_ref, an exact non-negative integer

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

            // The accumulator is never rescaled.  Instead it is pinned to a fixed
            // log2 frame m_ref (the first tile's max) and the per-tile power of
            // two rides the PV mfma's scale_b, which is free.  That removes 256
            // v_mul plus ~490 v_accvgpr moves per tile -- ~40% of all VALU.
            //
            // For the correction to be exactly representable in E8M0 the running
            // max is rounded to an integer.  Rounding *down* is what makes this
            // safe: P = 2^(s-new_m) is then <= 2, so after the +7 gain the fp8
            // operand peaks at 256 (e4m3 tops out at 448) and every value sits
            // one bit *further* from the subnormal cliff rather than one bit
            // closer, which is what a ceil() here costs.
            const float new_m = __builtin_floorf(max(m_row[qs], row_max));
            if (m_ref[qs] == opus::numeric_limits<float>::lowest()) {
                m_ref[qs] = new_m;
                // The sink joins l_row here, in the frame that is being fixed
                // right now, so m_ref never has to outlive the loop.  Letting it
                // reach the epilogue instead costs 1004 bytes/lane of scratch:
                // one extra live float per sub-tile is all it takes to tip the
                // allocator into evicting the whole accumulator.  The clamp only
                // fires when the sink dominates every logit by more than 2^96,
                // where the attention output is zero regardless.
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
            // The rope V reads are issued *before* this k-step's P is computed,
            // so the 8 ds_read_b64_tr_b16 hide behind ~60 VALU ops of exp2 /
            // ldexp / cvt instead of stalling in front of 8 bf16 mfmas.  There
            // is nothing else to cover them with: unlike the NoPE PV, which has
            // 28 output subtiles per drain, the rope PV has only 4.  Costs no
            // registers -- r was already live across the wait.  Worth ~4%.
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
                const int ew = __builtin_bit_cast(
                    int, s_etok.template _load<4>(16 * a + 4 * g));
                static_for<T::Q_SUB>([&](auto i_qs) {
                    constexpr int qs = i_qs.value;
                    vector_t<float, 4> p4, praw;
#pragma unroll
                    for (int b = 0; b < 4; ++b) {
                        const float p = __builtin_amdgcn_exp2f(
                            __builtin_fmaf(v_s[qs][a * 4 + b], c_row, bias[qs]));
                        praw[b] = p;
                        p4[b] = __builtin_ldexpf(
                            p, ((ew >> (8 * b)) & 0xff) - max_e_t + 7 - dexp[qs]);
                    }
                    rsum[qs] += (praw[0] + praw[1]) + (praw[2] + praw[3]);
                    const int hw = __builtin_bit_cast(int, cast<fp8_t>(p4));
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
                    v_o[qs][o0] = mma_bf16(vr, pr[qs], v_o[qs][o0]);
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

        // The partial lgkmcnt counts in the loop below assume the LDS queue
        // holds nothing but this loop's tr_loads.  _tr_load is inline asm the
        // compiler cannot see, so one compiler-emitted ds_read left in flight
        // makes lgkmcnt(8) return before this o_tile's reads have landed --
        // garbage V, NaN output, and only the NoPE columns affected (P and the
        // RoPE PV stay bit-exact, which is what pins the fault to here).  Three
        // unrelated scheduling changes each tripped it.  Pin every compiler LDS
        // op above this point, then drain, so the counts mean what they say.
        asm volatile("" ::: "memory");
        s_waitcnt_lgkmcnt(0_I);

        // V is fetched in batches of PV_BATCH o_tiles: issue every tr_load of
        // the batch, one full drain, then all its mfmas.
        //
        // The obvious alternative -- a rolling prefetch with a *partial*
        // lgkmcnt -- is unsafe here and was the source of three "miscompiles"
        // chased earlier.  lgkmcnt counts every LGKM op, and between two of this
        // loop's tr_loads the compiler happily places the next tile's QK
        // ds_read_b128s, the RoPE tr_b16s, LDS stores and even an s_load (SMEM,
        // which retires out of order).  `lgkmcnt(8)` then returns before this
        // o_tile's reads have landed, giving garbage V.  It only corrupts the
        // NoPE columns -- P and the RoPE PV stay bit-exact -- which is what
        // makes it look like a codegen bug rather than a counting bug.
        static_for<T::O_TILES / T::PV_BATCH>([&](auto i_b) {
            constexpr int b0 = i_b.value * T::PV_BATCH;
            i32x8 vv[T::PV_BATCH];
            static_for<T::PV_BATCH>([&](auto i_j) {
                constexpr int j = i_j.value;
                vv[j] = load_v(number<b0 + j>{});
            });
#if PA_PV_PARTIAL_WAIT
            // EXPERIMENT (see the block comment above): the batch's j-th mfma
            // only needs vv[j], and LDS retires in order, so waiting for
            // lgkmcnt(PV_BATCH-1-j) is sufficient *provided nothing but this
            // batch's tr_loads sits in the LGKM queue*.  The `asm memory`
            // barrier before the batch pins the compiler's own LDS ops above
            // it; an s_load landing between two tr_loads would still break it
            // (SMEM retires out of order) and that is exactly the failure the
            // comment above records three times.  Measured arm only.
            static_for<T::PV_BATCH>([&](auto i_j) {
                constexpr int j = i_j.value;
                s_waitcnt_lgkmcnt(number<T::PV_BATCH - 1 - j>{});
                asm volatile("" : "+v"(vv[j]) ::);
                static_for<T::Q_SUB>([&](auto i_qs) {
                    constexpr int qs = i_qs.value;
                    v_o[qs][b0 + j] =
                        mma_sb(vv[j], v_p[qs], v_o[qs][b0 + j], pv_e0 + dexp[qs]);
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
                        mma_sb(vv[j], v_p[qs], v_o[qs][b0 + j], pv_e0 + dexp[qs]);
                });
            });
#endif
        });

        if (has_next) {
#pragma unroll
            for (int i = 0; i < T::SLOTS_PER_WAVE; ++i) row_pf[i] = row_pf2[i];

            // The rope tile is single-buffered, so it can only be refilled once
            // every wave has finished its PV-RoPE reads.  Issuing it earlier (a
            // barrier right after softmax phase 2, so it hides under the NoPE
            // PV) was tried and is 7% *slower*: the copy's LDS writes contend
            // with the PV's LDS reads, and the mid-tile barrier stalls waves
            // that would otherwise have converged at the tile boundary anyway.
            if constexpr (T::ROPE_BUFS == 1) {
                // single-buffered: it can only be refilled once every wave has
                // finished reading it
                lds_barrier();
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
void pa_prefill_fp8_kernel(pa_fp8_kargs kargs)
{
    using namespace opus;
    using namespace pa_fp8;
    using T = pa_fp8_traits;

    const int q_token = kargs.N - 1 - (int)blockIdx.x;  // heavy tokens dispatch first
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
    // Q arrives as bf16 [N, H, 448] and is quantised here, in registers.  A
    // separate pack pass costs a flat ~35 us/call at T=1024 H=128 and moves
    // ~184 MB at ~5.3 TB/s -- it is at bandwidth, so the only way to remove it
    // is to never materialise a packed Q.  Worth 1.06-1.17x of the per-call
    // cost; the kernel gives some of it back reading bf16 (2x the bytes).
    //
    // One exponent per head, not per 32.  The mfma's E8M0 scale partition is
    // not lane-local -- a lane's byte governs 16 of its own elements plus 16 of
    // a neighbour's -- so an in-register per-32 pack would have to reproduce
    // that partition exactly; a uniform byte per head makes it irrelevant.
    // Verified accuracy-neutral: identical rel_l2 out to 25 octaves of forced
    // within-head spread, because elements that far below the head max
    // contribute less than the fp8 mantissa of the dominant terms.
    //
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
#pragma unroll
    for (int qs = 0; qs < T::Q_SUB; ++qs)
#pragma unroll
        for (int j = 0; j < T::O_LEN; ++j) v_o[qs][j] = f32x4{0.f, 0.f, 0.f, 0.f};
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
    // Coalesced epilogue -- see the PA_EPI_LDS note in pa_fp8_traits for the
    // measurements.  1.047x at the c4 shape, 1.083x at c128, 1.099x synthetic,
    // 1.067x/1.085x at N=4096, results bit-identical.
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
            static_for<T::D_TILE / 16>([&](auto i_ot) {
                constexpr int ot = i_ot.value;
                // the RoPE columns never went through the P gain, so they must
                // not be divided by it
                const float os = (ot < T::O_TILES) ? (o_scale * inv_alpha)
                                                   : o_scale;
                s_ep.template store<4>(
                    cast<bf16_t>(__builtin_bit_cast(vector_t<float, 4>,
                                                    v_o[qs][ot] * os)),
                    c * EP_PITCH + ot * 16 + g * 4);
            });
            // The 16 rows a wave reads back are the 16 it just wrote, so this
            // only has to order a wave against itself -- lgkmcnt, not a barrier.
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
                store<8>(g_o, s_ep.template load<8>(hh * EP_PITCH + lane * 8),
                         (h_wave + qs * T::Q_TILE + hh) * kargs.stride_o_h
                             + lane * 8, 0, number<PA_NT_AUXO>{});
            }
            s_waitcnt_lgkmcnt(0_I);   // the rows are reused by the next qs
        }
    }
#else
#pragma unroll
    for (int qs = 0; qs < T::Q_SUB; ++qs) {
        const float o_scale = (l_row[qs] > 0.f) ? (1.f / l_row[qs]) : 0.f;
        const int o_off = (h_wave + qs * T::Q_TILE + c) * kargs.stride_o_h + 4 * g;

        static_for<T::D_TILE / 16>([&](auto i_ot) {   // 28 NoPE + 4 RoPE subtiles
            constexpr int ot = i_ot.value;
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

}  // namespace pa_fp8_h40

// Q packing is included at *file* scope, not from inside namespace pa_fp8_h40:
// pa_fp8_q_pack.h opens that namespace itself, so including it one level in
// would define everything as pa_fp8_h40::pa_fp8_h40::*.  The host pass gets a
// stub so the launch site still compiles.
#if !defined(__HIP_DEVICE_COMPILE__) || !defined(__gfx950__)
namespace pa_fp8_h40 {
__global__ void pa_fp8_q_pack_kernel(const __bf16*, unsigned char*, int, int) {}
}  // namespace pa_fp8_h40
#else
#include "pa_fp8_q_pack.h"
#endif
