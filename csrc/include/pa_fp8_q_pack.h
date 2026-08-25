// ===========================================================================
// Q packing for the DSv4 MLA sparse prefill kernel
// ===========================================================================
//
// The prefill kernel packs Q in its own prologue; this standalone pass is for
// callers that want the packed buffer for something else.
#pragma once

namespace pa_fp8_h40 {

// ---------------------------------------------------------------------------
// Q packing: bf16 [rows, 448] -> the op's packed 512-byte row
//   448 NoPE fp8 | 14 per-32 UE8M0 | 50 pad
// Promoted from the dev harness (dev_main_dsa.hip:30 pack_rows), which is the
// reference this contract was validated against.  One block per row, one
// thread per 32-element quant block.
//
// The exponent is ceil(log2(amax/240)) + 127 rather than /448: 240 leaves the
// headroom the PV's P-gain alpha assumes (see the scale-routing note in
// pa_sparse_prefill_fp8_h40.h).  The pad is zeroed so the row is deterministic;
// the QK never reads d >= 448 on the Q side, so its content is don't-care.
// ---------------------------------------------------------------------------
constexpr int QPACK_D_NOPE = 448;
constexpr int QPACK_NBLK   = QPACK_D_NOPE / 32;   // 14
constexpr int QPACK_ROW    = 512;

__global__ void pa_fp8_q_pack_kernel(const __bf16* __restrict__ src,
                                     unsigned char* __restrict__ dst,
                                     int rows,
                                     int src_stride)
{
    const int row = (int)blockIdx.x;
    if(row >= rows)
        return;
    const int blk = (int)threadIdx.x;

    unsigned char* const d_row = dst + (size_t)row * QPACK_ROW;
    if(blk >= QPACK_NBLK)
    {
        // zero the 50 pad bytes with the threads that have no quant block
        const int t = blk - QPACK_NBLK;
        if(t < QPACK_ROW - QPACK_D_NOPE - QPACK_NBLK)
            d_row[QPACK_D_NOPE + QPACK_NBLK + t] = 0;
        return;
    }

    // The 32 elements are read once as 16 dwords and kept in registers: both
    // the absmax pass and the conversion pass below use them, so the second
    // pass costs no loads.
    //
    // absmax runs in the *integer* domain.  bf16 is the top half of an fp32, so
    // with the sign masked off its bit pattern orders exactly like the
    // magnitude -- no bf16->f32 conversion is needed to find the largest.  Two
    // elements share a dword, so one v_pk_max_u16 covers a pair.
    // (Differs from fmaxf on NaN, which fmaxf would ignore and this picks;
    // post-norm q never carries one, and a NaN here would be a bug upstream.)
    typedef unsigned short u16x2 __attribute__((ext_vector_type(2)));
    const unsigned* const sw = reinterpret_cast<const unsigned*>(
        src + (size_t)row * src_stride + blk * 32);
    unsigned raw[16];
    u16x2 acc = {0, 0};
#pragma unroll
    for(int i = 0; i < 16; ++i)
    {
        raw[i] = sw[i];
        acc    = __builtin_elementwise_max(acc,
                     __builtin_bit_cast(u16x2, raw[i] & 0x7FFF7FFFu));
    }
    const unsigned mx   = acc.x > acc.y ? acc.x : acc.y;
    const float    amax = __uint_as_float(mx << 16);

    // ceil(log2(y)) straight off the float's exponent field, and 2^-E built by
    // planting an exponent -- no log2f/ceilf/exp2f.  For y > 0 with exponent
    // field ex and mantissa mf:  y == 2^(ex-127) exactly when mf == 0, so
    // ceil(log2 y) = (ex-127) + (mf != 0), and e = that + 127 = ex + (mf != 0).
    // Denormal (ex==0) and inf (ex==255) both fall into the clamp below.
    //
    // This is also *more* correct than the transcendental form it replaces:
    // v_log_f32 is a ~1 ULP approximation, so ceilf(log2f(y)) can come out one
    // too high on an exact power of two (harmless, costs a mantissa bit) or one
    // too low just under a boundary (not harmless -- amax*inv can then exceed
    // e4m3's 448 and clamp).  The bit form cannot do either.
    int e = 127;
    if(amax > 0.f)
    {
        const unsigned bits = __float_as_uint(amax / 240.0f);
        e = (int)((bits >> 23) & 0xFFu) + (int)((bits & 0x7FFFFFu) != 0u);
        e = e < 1 ? 1 : (e > 200 ? 200 : e);
    }
    d_row[QPACK_D_NOPE + blk] = (unsigned char)e;

    // The *decode* scale 2^(e-127), i.e. bit pattern e<<23.  MX convention:
    // v_cvt_scalef32_* takes the factor that will be applied when reading the
    // fp8 back, and divides by it on the way in -- so this is the reciprocal of
    // the 2^(127-e) the old multiply-then-convert form used.  e in [1,200]
    // keeps the planted exponent field normal.
    const float dec = __uint_as_float((unsigned)e << 23);
    // cvt_pk_fp8_f32 converts and packs *two* floats per instruction, and the
    // word_sel operand picks which half of the destination dword it lands in --
    // so four elements become two instructions and one dword store.  Feeding it
    // a 0.0f second operand and taking only the low byte, as this used to,
    // threw away half the conversion throughput and turned every element into
    // its own byte store.  blk*32 keeps d 32-byte aligned, so the dword stores
    // are safe.  Conversion is elementwise, so the bytes are unchanged.
    // gfx950's v_cvt_scalef32_pk_fp8_bf16 takes the bf16 pair *and* an fp32
    // scale and emits the packed fp8 pair, so the widen / multiply / convert
    // chain collapses into one instruction per two elements.  inv is exactly
    // 2^(127-e), whose bit pattern is (254-e)<<23 -- the E8M0 form this
    // instruction family expects -- and scaling by a power of two is exact, so
    // there is still only the one rounding at the fp8 conversion and the bytes
    // are unchanged.  word_sel picks which half of the dword the pair lands in.
    typedef short  s16x2  __attribute__((ext_vector_type(2)));
    typedef __bf16 bf16x2 __attribute__((ext_vector_type(2)));
    unsigned* const dw = reinterpret_cast<unsigned*>(d_row + blk * 32);
#pragma unroll
    for(int j = 0; j < 16; j += 2)
    {
        s16x2 w = {0, 0};
        w = __builtin_amdgcn_cvt_scalef32_pk_fp8_bf16(
                w, __builtin_bit_cast(bf16x2, raw[j]), dec, false);
        w = __builtin_amdgcn_cvt_scalef32_pk_fp8_bf16(
                w, __builtin_bit_cast(bf16x2, raw[j + 1]), dec, true);
        dw[j >> 1] = __builtin_bit_cast(unsigned, w);
    }
}

// REJECTED, measured twice (2026-08-03): a coalesced rewrite -- one lane per
// *element*, 7 passes of 64 contiguous bf16, per-32 absmax by an xor reduction
// -- is **12% slower** than the scattered form above, byte-identical either way:
//
//     scalar + transcendental   635 us      (T=8192 H=128, kernel only)
//     scalar + bit math         594 us   <- shipped
//     coalesced + transcendental 711 us
//     coalesced + bit math      717 us
//
// The first attempt was blamed on the rewrite evaluating log2f/ceilf/exp2f in
// every lane on all 7 passes (21 instruction issues per row against 3).  That
// was wrong: with the exponent math reduced to integer ops the rewrite is still
// 717 us.  What is left is the 35 cross-lane ops per row and the 7 x 64-byte
// stores (one byte per lane, half a cache line each).  Fixing the access
// pattern is not worth what the reduction costs -- 14 lanes each streaming 32
// contiguous elements is cheaper than 64 lanes cooperating.
//
// Do not retry without first measuring where this kernel actually stalls; the
// 2.3 TB/s it achieves against a 5.6 TB/s pure-copy is *not* evidence that it
// is bandwidth-bound.

}  // namespace pa_fp8_h40
