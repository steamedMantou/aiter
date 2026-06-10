import torch
import triton
import triton.language as tl

from aiter import dtypes
from typing import Tuple
# FP4 E2M1 format constants
# Representable absolute values: 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
_FP8_MAX = 448.0
FP4_MAX = 6.0
FP4_BLOCK_DIM = 32  # quantization block size along head_dim

def cdiv(m, n):
    return (m - 1) // n + 1

def _block_quantize_fp4_prune_config(configs, named_args, **kwargs):
    QUANT_GRANULARITY = kwargs['QUANT_GRANULARITY']
    pruned_configs = []
    for config in configs:
        block_size = config.kwargs['BLOCK_SIZE']
        if block_size % QUANT_GRANULARITY == 0:
            pruned_configs.append(config)
    return pruned_configs


def _block_quantize_kernel_prune_config(configs, named_args, **kwargs):
    # 1. Extract the runtime value of n_cols
    QUANT_GRANULARITY = kwargs['QUANT_GRANULARITY']
    pruned_configs = []

    for config in configs:
        block_size = config.kwargs['BLOCK_SIZE']
        if block_size % QUANT_GRANULARITY == 0:
            pruned_configs.append(config)
    return pruned_configs


@triton.jit
def _round_to_fp4_code(abs_val):
    code = (abs_val * 0.0).to(tl.int32)           
    code = code + (abs_val > 0.25).to(tl.int32)   
    code += (abs_val >= 0.75).to(tl.int32)
    code = code + (abs_val > 1.25).to(tl.int32)   
    code += (abs_val >= 1.75).to(tl.int32)   
    code = code + (abs_val > 2.5).to(tl.int32)    
    code += (abs_val >= 3.5).to(tl.int32)    
    code = code + (abs_val > 5.0).to(tl.int32)    
    return code 


@triton.jit
def _compute_e8m0_scale(abs_max, EPS):
    safe_amax = tl.where(abs_max > 0.0, abs_max, 1.0)
    log2_val = tl.log2(safe_amax)

    floored = tl.math.floor(log2_val)
    frac = log2_val - floored
    rounded = tl.where(frac > 0.5, floored + 1.0, floored)
    is_tie = (frac == 0.5)
    floored_int = floored.to(tl.int32)
    floor_is_even = ((floored_int & 1) == 0)
    rounded = tl.where(is_tie & floor_is_even, floored, rounded)
    rounded = tl.where(is_tie & ~floor_is_even, floored + 1.0, rounded)

    e = tl.minimum(tl.maximum(rounded, -127.0), 127.0)
    e = tl.where(abs_max > 0.0, e, 2.0)
    
    scale_float = tl.math.exp2(e - 2.0)
    #scale_e8m0 = (e + 125.0).to(tl.int32)
    scale_e8m0_raw = (e + 125.0).to(tl.int32)
    scale_e8m0 = tl.minimum(tl.maximum(scale_e8m0_raw, 0), 254)
    #print("_compute_e8m0_scale scale_float", scale_float)
    #print("_compute_e8m0_scale scale_e8m0", scale_e8m0)
    return scale_float, scale_e8m0


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': BS, "waves_per_eu": wpe}, num_stages=ns, num_warps=nw)
        for BS in [16, 64]
        for nw in [1, 4]
        for ns in [1]
        for wpe in [1, 2]
    ],
    key=['HDIMS', 'QUANT_GRANULARITY'],
    prune_configs_by={'early_config_prune': _block_quantize_fp4_prune_config}
)
@triton.jit
def block_quantize_fp4_kernel(
    Input, Quant_input, Scale, seqlen_csum_cu,
    input_stride_seq, input_stride_head, quant_input_stride_seq, quant_input_stride_head,
    scale_stride_head, scale_stride_seq,
    EPS: tl.constexpr, HDIMS: tl.constexpr, FP4_BLOCK_DIM: tl.constexpr,     
    QUANT_GRANULARITY: tl.constexpr, BLOCK_SIZE: tl.constexpr, FP4_MAX: tl.constexpr,           
):
    block_id = tl.program_id(0)
    head_id = tl.program_id(1)
    batch_id = tl.program_id(2)

    block_id_i64 = block_id.to(tl.int64)
    head_id_i64 = head_id.to(tl.int64)
    input_stride_seq_i64 = input_stride_seq.to(tl.int64)
    input_stride_head_i64 = input_stride_head.to(tl.int64)
    quant_input_stride_seq_i64 = quant_input_stride_seq.to(tl.int64)
    quant_input_stride_head_i64 = quant_input_stride_head.to(tl.int64)
    scale_stride_head_i64 = scale_stride_head.to(tl.int64)
    scale_stride_seq_i64 = scale_stride_seq.to(tl.int64)

    seq_off = block_id_i64 * BLOCK_SIZE
    seq_start = tl.load(seqlen_csum_cu + batch_id).to(tl.int64)
    seq_end = tl.load(seqlen_csum_cu + batch_id + 1).to(tl.int64)

    if seq_start + seq_off >= seq_end:
        return

    input_base = Input + (seq_start + seq_off) * input_stride_seq_i64 + head_id_i64 * input_stride_head_i64
    quant_base = Quant_input + (seq_start + seq_off) * quant_input_stride_seq_i64 + head_id_i64 * quant_input_stride_head_i64
    # Scale 张量严格和 seq 维度一一映射
    scale_base = Scale + head_id_i64 * scale_stride_head_i64 + (seq_start + seq_off) * scale_stride_seq_i64

    block_range = tl.minimum(BLOCK_SIZE, seq_end - seq_start - seq_off)
    NUM_HDIM_BLOCKS: tl.constexpr = HDIMS // FP4_BLOCK_DIM
    HALF_BLOCK: tl.constexpr = FP4_BLOCK_DIM // 2  

    seq_offs = tl.arange(0, QUANT_GRANULARITY).to(tl.int64)
    pair_offs = tl.arange(0, HALF_BLOCK).to(tl.int64)

    for i in range(0, BLOCK_SIZE, QUANT_GRANULARITY):
        mask_seq = (i + seq_offs) < block_range

        for hb in tl.static_range(NUM_HDIM_BLOCKS):
            hdim_base = hb * FP4_BLOCK_DIM

            even_ptrs = input_base + (i + seq_offs)[:, None] * input_stride_seq_i64 + (hdim_base + pair_offs * 2)[None, :]
            even_vals = tl.load(even_ptrs, mask=mask_seq[:, None], other=0.0).to(tl.float32)

            odd_ptrs = input_base + (i + seq_offs)[:, None] * input_stride_seq_i64 + (hdim_base + pair_offs * 2 + 1)[None, :]
            odd_vals = tl.load(odd_ptrs, mask=mask_seq[:, None], other=0.0).to(tl.float32)

            abs_max = tl.maximum(tl.max(tl.abs(even_vals), 1), tl.max(tl.abs(odd_vals), 1))

            scale_float, scale_e8m0 = _compute_e8m0_scale(abs_max, EPS)
            

            even_scaled = tl.minimum(tl.abs(even_vals / scale_float[:, None]), FP4_MAX)
            even_code = _round_to_fp4_code(even_scaled)            
            even_sign = (even_vals < 0.0).to(tl.int32) * 8         
            even_fp4 = even_code | even_sign                        

            odd_scaled = tl.minimum(tl.abs(odd_vals / scale_float[:, None]), FP4_MAX)
            odd_code = _round_to_fp4_code(odd_scaled)
            odd_sign = (odd_vals < 0.0).to(tl.int32) * 8
            odd_fp4 = odd_code | odd_sign

            packed = ((odd_fp4 << 4) | even_fp4).to(tl.uint8)

            out_ptrs = quant_base + (i + seq_offs)[:, None] * quant_input_stride_seq_i64 + (hb * HALF_BLOCK + pair_offs)[None, :]
            tl.store(out_ptrs, packed, mask=mask_seq[:, None])

            scale_ptrs = scale_base + (i + seq_offs) * scale_stride_seq_i64 + hb
            tl.store(scale_ptrs, scale_e8m0.to(tl.uint8), mask=mask_seq)

def block_quantize_fp4(
    input: torch.Tensor,
    seqlen_csum_cu: torch.Tensor,
    seqlens_list: list[int],
    granularity: int,
    quant_dtype: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    assert sum(seqlens_list) == input.shape[1]
    assert granularity < 128
    assert input.stride(-1) == 1

    batch = len(seqlens_list)
    total_seqlens = sum(seqlens_list)
    num_head = input.shape[0]
    hdims = input.shape[2]
    max_seqlen = max(seqlens_list)
    assert hdims % FP4_BLOCK_DIM == 0

    num_hdim_blocks = hdims // FP4_BLOCK_DIM

    quant_input = torch.empty(num_head, total_seqlens, hdims // 2, dtype=torch.uint8, device=input.device)
    scale = torch.empty(num_head, total_seqlens, num_hdim_blocks, dtype=torch.uint8, device=input.device)
    scale_len_csum_cu = seqlen_csum_cu.clone()

    grid = lambda META: (triton.cdiv(max_seqlen, META['BLOCK_SIZE']), num_head, batch)

    block_quantize_fp4_kernel[grid](
        input, quant_input, scale, seqlen_csum_cu,
        input.stride(1), input.stride(0),
        quant_input.stride(1), quant_input.stride(0),
        scale.stride(0), scale.stride(1),
        EPS=torch.finfo(torch.float32).eps,
        HDIMS=hdims, FP4_BLOCK_DIM=FP4_BLOCK_DIM,
        QUANT_GRANULARITY=granularity, FP4_MAX=FP4_MAX,
    )

    return quant_input, scale, scale_len_csum_cu


@triton.autotune(
    configs=[
        triton.Config(
            {'BLOCK_SIZE': BS, "waves_per_eu": wpe},
            num_stages=ns,
            num_warps=nw
        )
        for BS in [16, 64]
        for nw in [1, 4]
        for ns in [1]
        for wpe in [1, 2]
    ],
    key=['hdims', 'QUANT_GRANULARITY'],
    prune_configs_by={
        'early_config_prune': _block_quantize_kernel_prune_config,
    }
)
@triton.jit
def block_quantize_kernel(
    Input,
    Quant_input,
    Scale,
    seqlen_csum_cu,
    scale_len_csum_cu,
    input_stride_seq,
    input_stride_head,
    quant_input_stride_seq,
    quant_input_stride_head,
    scale_stride_head,
    EPS: tl.constexpr,
    HDIMS: tl.constexpr,
    QUANT_GRANULARITY: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    block_id = tl.program_id(0)
    head_id = tl.program_id(1)
    batch_id = tl.program_id(2)
    block_id_i64 = block_id.to(tl.int64)
    head_id_i64 = head_id.to(tl.int64)
    input_stride_seq_i64 = input_stride_seq.to(tl.int64)
    input_stride_head_i64 = input_stride_head.to(tl.int64)
    quant_input_stride_seq_i64 = quant_input_stride_seq.to(tl.int64)
    quant_input_stride_head_i64 = quant_input_stride_head.to(tl.int64)
    scale_stride_head_i64 = scale_stride_head.to(tl.int64)

    seq_off = block_id_i64 * BLOCK_SIZE
    seq_start = tl.load(seqlen_csum_cu + batch_id).to(tl.int64)
    seq_end = tl.load(seqlen_csum_cu + batch_id + 1).to(tl.int64)

    # Just for small batch_size
    scale_seq_start = tl.full((), 0, dtype=tl.int64)
    for i in range(batch_id):
        cur_seq_start = tl.load(seqlen_csum_cu + i).to(tl.int64)
        cur_seq_end = tl.load(seqlen_csum_cu + i + 1).to(tl.int64)
        cur_seq_len = cur_seq_end - cur_seq_start
        scale_seq_start = scale_seq_start + tl.cdiv(cur_seq_len, QUANT_GRANULARITY)
    scale_seq_end = scale_seq_start + tl.cdiv(seq_end - seq_start, QUANT_GRANULARITY)
    # If calculate scale_len_csum_cu outsize to prevent range(batch_id)
    # scale_seq_start = tl.load(scale_len_csum_cu + batch_id)

    quant_type: tl.constexpr = Quant_input.type.element_ty

    if seq_start + seq_off >= seq_end:
        return

    input_ptr = Input + (seq_start + seq_off) * input_stride_seq_i64 + head_id_i64 * input_stride_head_i64
    quant_input_ptr = Quant_input + (seq_start + seq_off) * quant_input_stride_seq_i64 + head_id_i64 * quant_input_stride_head_i64

    scale_ptr = Scale + head_id_i64 * scale_stride_head_i64 + scale_seq_start + block_id_i64 * BLOCK_SIZE // QUANT_GRANULARITY
    block_range = tl.minimum(BLOCK_SIZE, seq_end - seq_start - seq_off)

    quant_offs = tl.arange(0, QUANT_GRANULARITY).to(tl.int64)
    hdim_offs = tl.arange(0, HDIMS).to(tl.int64)
    for i in range(0, BLOCK_SIZE, QUANT_GRANULARITY):
        input_ptrs = input_ptr + (i + quant_offs)[:, None] * input_stride_seq_i64 + hdim_offs[None, :]
        mask_seq = (i + quant_offs) < block_range
        mask_scale = i < block_range
        vals = tl.load(input_ptrs, mask=mask_seq[:, None], other=0.0).to(tl.float32)

        #  计算动态量化 scale
        scale = tl.max(tl.abs(vals))
        if quant_type == tl.int8:
            scale = scale / 127.0
        elif quant_type == tl.float8e4nv:
            scale = scale / 448.0
        elif quant_type == tl.float8e4b8:   # torch.float8_e4m3fnuz
            scale = scale / 240
        safe_scale = tl.where(scale < EPS, EPS, scale)
        quant_vals = vals / safe_scale
        if quant_type == tl.int8:
            quant_vals += 0.5 * tl.where(quant_vals >= 0, 1.0, -1.0)

        quant_vals = quant_vals.to(quant_type)
        quant_input_ptrs = quant_input_ptr + (i + quant_offs)[:, None] * quant_input_stride_seq_i64 + hdim_offs[None, :]
        tl.store(quant_input_ptrs, quant_vals, mask=mask_seq[:, None])
        tl.store(scale_ptr + (i // QUANT_GRANULARITY), scale, mask=mask_scale)

    if block_id == 0 and head_id == 0:
        tl.store(scale_len_csum_cu + batch_id + 1, scale_seq_end)
        if batch_id == 0:
            tl.store(scale_len_csum_cu, 0)

# 是 INT8 动态量化，per_head_token 级别的量化
def block_quantize(
    input: torch.Tensor,
    seqlen_csum_cu: torch.Tensor,
    seqlens_list: list[int],
    granularity: int,
    quant_dtype: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Args:
        input: (num_heads, seq_len, head_dim)
        seqlen_csum_cu: (seq_len + 1,)
        seqlens_list: (seq_len,)
        granularity: smaller than 128
        quant_dtype: i8 or fp8
    
    Return:
        quant_input: (num_heads, seq_len, head_dim)
        scale: (num_head, seq_len)
        scale_len_csum_cu: (seq_len + 1,)
    """
    quant_dtype = dtypes.str2Dtype(quant_dtype)
    assert quant_dtype in [dtypes.i8, dtypes.fp8]
    assert sum(seqlens_list) == input.shape[1]
    assert granularity < 128
    assert input.stride(-1) == 1, "Input tensor must be contiguous in the last dimension"

    batch = len(seqlens_list)
    total_seqlens = sum(seqlens_list)
    num_head = input.shape[0]
    hdims = input.shape[2]
    max_seqlen = max(seqlens_list)

    scale_seqlens_list = [cdiv(seqlen, granularity) for seqlen in seqlens_list]
    total_num_scale = sum(scale_seqlens_list)
    # scale_len_csum_cu = torch.tensor([0] + scale_seqlens_list, dtype=torch.int32, device=input.device).cumsum(dim=0, dtype=torch.int32)

    quant_input = torch.empty(num_head, total_seqlens, hdims, dtype=quant_dtype, device=input.device)

    scale = torch.empty(num_head, total_num_scale, dtype=torch.float32, device=input.device)
    scale_len_csum_cu = torch.empty(batch + 1, dtype=torch.int32, device=input.device)

    # block_size = 64 if granularity < 128 else 128
    # grid = (cdiv(max_seqlen, block_size), num_head, batch)
    grid = lambda META: (
        triton.cdiv(max_seqlen, META['BLOCK_SIZE']), 
        num_head, 
        batch
    )
    block_quantize_kernel[grid](
        input,
        quant_input,
        scale,
        seqlen_csum_cu,
        scale_len_csum_cu,
        input.stride(1),
        input.stride(0),
        quant_input.stride(1),
        quant_input.stride(0),
        scale.stride(0),
        EPS=torch.finfo(torch.float32).eps,
        HDIMS=hdims,
        QUANT_GRANULARITY=granularity,
        # BLOCK_SIZE=block_size,
    )

    return quant_input, scale, scale_len_csum_cu


@triton.jit
def _quantize_fp8_e8m0_vec_kernel(
    X_ptr,
    OUT_ptr,
    SCALE_ptr,
    stride_row,
    N: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    FP8_MAX: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,  # number of groups per tile (process multiple groups per iteration)
):
    """
    Vectorized: each program handles one row.
    Tiles BLOCK_GROUPS groups (= BLOCK_GROUPS * GROUP_SIZE elements) per iteration.
    Grid: (num_rows,)
    """
    pid_row = tl.program_id(0).to(tl.int64)
    row_base = pid_row * stride_row

    TILE_SIZE: tl.constexpr = BLOCK_GROUPS * GROUP_SIZE

    for tile_start in range(0, NUM_GROUPS, BLOCK_GROUPS):
        # Group offsets in this tile
        g_offs = (tile_start + tl.arange(0, BLOCK_GROUPS)).to(tl.int64)
        g_mask = g_offs < NUM_GROUPS

        # Element offsets: [BLOCK_GROUPS, GROUP_SIZE]
        elem_base = g_offs * GROUP_SIZE  # [BLOCK_GROUPS]
        elem_inner = tl.arange(0, GROUP_SIZE).to(tl.int64)  # [GROUP_SIZE]
        elem_offs = elem_base[:, None] + elem_inner[None, :]  # [BLOCK_GROUPS, GROUP_SIZE]
        elem_mask = g_mask[:, None] & (elem_offs < N)

        # Load [BLOCK_GROUPS, GROUP_SIZE]
        x_ptrs = X_ptr + row_base + elem_offs
        x = tl.load(x_ptrs, mask=elem_mask, other=0.0).to(tl.float32)

        # Per-group amax: reduce over GROUP_SIZE dim → [BLOCK_GROUPS]
        amax = tl.max(tl.abs(x), axis=1)
        amax = tl.maximum(amax, 1e-30)

        # E8M0 exponent, optimized without log2/floor/exp2.
        # Need exp = ceil(log2(amax / 448)). Since 448 = 1.75 * 2^8,
        # exp = floor_log2(amax) - 8 + (mantissa(amax) > 1.75).
        amax_bits = amax.to(tl.int32, bitcast=True)
        exp_i32 = ((amax_bits >> 23) & 0xFF) - 127
        mantissa = amax_bits & 0x7FFFFF
        exp_i32 = exp_i32 - 8 + (mantissa > 0x600000).to(tl.int32)
        exp_i32 = tl.minimum(tl.maximum(exp_i32, -127), 127)

        # Construct scale = 2^exp by writing the fp32 exponent bits directly.
        scale_byte_i32 = exp_i32 + 127
        scale_bits = scale_byte_i32 << 23
        scale = scale_bits.to(tl.float32, bitcast=True)  # [BLOCK_GROUPS]

        # Normalize: broadcast scale over GROUP_SIZE
        x_norm = x / scale[:, None]
        x_norm = tl.minimum(tl.maximum(x_norm, -FP8_MAX), FP8_MAX)

        # Cast to fp8
        x_fp8 = x_norm.to(tl.float8e4nv)

        # Store fp8 output
        out_ptrs = OUT_ptr + row_base + elem_offs
        tl.store(out_ptrs, x_fp8, mask=elem_mask)

        # Store scale bytes
        scale_byte = tl.minimum(tl.maximum(scale_byte_i32, 0), 255).to(tl.uint8)
        scale_ptrs = SCALE_ptr + pid_row * NUM_GROUPS + g_offs
        tl.store(scale_ptrs, scale_byte, mask=g_mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_ROWS": 4}, num_warps=4),
        triton.Config({"BLOCK_ROWS": 8}, num_warps=4),
        triton.Config({"BLOCK_ROWS": 16}, num_warps=4),
    ],
    key=["N", "GROUP_SIZE"],
)
@triton.jit
def _quantize_fp8_e8m0_rowblock_kernel(
    X_ptr,
    OUT_ptr,
    SCALE_ptr,
    stride_row,
    NUM_ROWS: tl.constexpr,
    N: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    FP8_MAX: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    """
    Row-batched FP8 path.
    One program handles multiple rows and all groups in each row to reduce scheduling
    overhead on long sequences. Supports GROUP_SIZE == N and smaller groups such as 32.
    """
    pid = tl.program_id(0)
    row_group_offs = tl.arange(0, BLOCK_ROWS * NUM_GROUPS).to(tl.int64)
    row_offs = (pid * BLOCK_ROWS + row_group_offs // NUM_GROUPS).to(tl.int64)
    group_offs = (row_group_offs % NUM_GROUPS).to(tl.int64)
    elem_inner = tl.arange(0, GROUP_SIZE).to(tl.int64)
    row_mask = row_offs < NUM_ROWS

    elem_offs = group_offs[:, None] * GROUP_SIZE + elem_inner[None, :]
    ptrs = X_ptr + row_offs[:, None] * stride_row + elem_offs
    x = tl.load(ptrs, mask=row_mask[:, None], other=0.0).to(tl.float32)

    amax = tl.max(tl.abs(x), axis=1)
    amax = tl.maximum(amax, 1e-30)

    amax_bits = amax.to(tl.int32, bitcast=True)
    exp_i32 = ((amax_bits >> 23) & 0xFF) - 127
    mantissa = amax_bits & 0x7FFFFF
    exp_i32 = exp_i32 - 8 + (mantissa > 0x600000).to(tl.int32)
    exp_i32 = tl.minimum(tl.maximum(exp_i32, -127), 127)

    scale_byte_i32 = exp_i32 + 127
    scale_bits = scale_byte_i32 << 23
    scale = scale_bits.to(tl.float32, bitcast=True)

    x_norm = x / scale[:, None]
    x_norm = tl.minimum(tl.maximum(x_norm, -FP8_MAX), FP8_MAX)
    x_fp8 = x_norm.to(tl.float8e4nv)

    out_ptrs = OUT_ptr + row_offs[:, None] * stride_row + elem_offs
    tl.store(out_ptrs, x_fp8, mask=row_mask[:, None])

    scale_byte = tl.minimum(tl.maximum(scale_byte_i32, 0), 255).to(tl.uint8)
    tl.store(SCALE_ptr + row_offs * NUM_GROUPS + group_offs, scale_byte, mask=row_mask)

@triton.jit
def _fp4_perchannel_kblock_kernel_v2(
    V_ptr,
    PACKED_ptr,
    SCALE_ptr,
    stride_v_bh: tl.int64,
    stride_v_t: tl.int64,
    stride_v_d: tl.int64,
    stride_p_bh: tl.int64,
    stride_p_t: tl.int64,
    stride_p_dhalf: tl.int64,
    stride_s_bh: tl.int64,
    stride_s_tb: tl.int64,
    stride_s_d: tl.int64,
    T: tl.int32,
    D: tl.constexpr,
    KBLOCK: tl.constexpr,
    BLOCK_D_HALF: tl.constexpr,  # number of D-pairs per tile (= BLOCK_D // 2)
):
    """
    Each program: one (bh, t_block) × BLOCK_D_HALF channel-pairs.
    Loads even+odd channels, quantizes both, packs into one byte.

    Grid: (BH * num_tblocks, cdiv(D//2, BLOCK_D_HALF))
    """
    pid_0 = tl.program_id(0).to(tl.int64)
    pid_1 = tl.program_id(1).to(tl.int64)

    num_tblocks = (T // KBLOCK).to(tl.int64)
    bh_idx = pid_0 // num_tblocks
    tb_idx = pid_0 % num_tblocks

    t_start = tb_idx * KBLOCK
    t_offs = (t_start + tl.arange(0, KBLOCK)).to(tl.int64)  # [KBLOCK]

    # D-pair offsets
    dp_start = pid_1 * BLOCK_D_HALF
    dp_offs = (dp_start + tl.arange(0, BLOCK_D_HALF)).to(tl.int64)  # [BLOCK_D_HALF]
    d_even = dp_offs * 2       # even channel indices
    d_odd = dp_offs * 2 + 1    # odd channel indices
    dp_mask = d_odd < D        # if odd is valid, even is too

    FP4_MAX: tl.constexpr = 6.0

    # ---- Load even channels: [KBLOCK, BLOCK_D_HALF] ----
    ptrs_even = (V_ptr + bh_idx * stride_v_bh
                 + t_offs[:, None] * stride_v_t
                 + d_even[None, :] * stride_v_d)
    v_even = tl.load(ptrs_even, mask=dp_mask[None, :], other=0.0).to(tl.float32)

    # ---- Load odd channels: [KBLOCK, BLOCK_D_HALF] ----
    ptrs_odd = (V_ptr + bh_idx * stride_v_bh
                + t_offs[:, None] * stride_v_t
                + d_odd[None, :] * stride_v_d)
    v_odd = tl.load(ptrs_odd, mask=dp_mask[None, :], other=0.0).to(tl.float32)

    # ---- Per-channel amax (over KBLOCK tokens): [BLOCK_D_HALF] each ----
    amax_even = tl.max(tl.abs(v_even), axis=0)
    amax_even = tl.maximum(amax_even, 1e-30)
    amax_odd = tl.max(tl.abs(v_odd), axis=0)
    amax_odd = tl.maximum(amax_odd, 1e-30)

    # ---- E8M0 exponent, optimized without log2/floor/exp2 ----
    # Need exp = ceil(log2(amax / 6)). Since 6 = 1.5 * 2^2,
    # exp = floor_log2(amax) - 2 + (mantissa(amax) > 1.5).
    bits_even = amax_even.to(tl.int32, bitcast=True)
    exp_even_i32 = ((bits_even >> 23) & 0xFF) - 127
    mant_even = bits_even & 0x7FFFFF
    exp_even_i32 = exp_even_i32 - 2 + (mant_even > 0x400000).to(tl.int32)
    exp_even_i32 = tl.minimum(tl.maximum(exp_even_i32, -127), 127)

    bits_odd = amax_odd.to(tl.int32, bitcast=True)
    exp_odd_i32 = ((bits_odd >> 23) & 0xFF) - 127
    mant_odd = bits_odd & 0x7FFFFF
    exp_odd_i32 = exp_odd_i32 - 2 + (mant_odd > 0x400000).to(tl.int32)
    exp_odd_i32 = tl.minimum(tl.maximum(exp_odd_i32, -127), 127)

    sb_even_i32 = exp_even_i32 + 127
    sb_odd_i32 = exp_odd_i32 + 127
    scale_even = (sb_even_i32 << 23).to(tl.float32, bitcast=True)  # [BLOCK_D_HALF]
    scale_odd = (sb_odd_i32 << 23).to(tl.float32, bitcast=True)    # [BLOCK_D_HALF]

    # ---- Store scales ----
    sb_even = tl.minimum(tl.maximum(sb_even_i32, 0), 255).to(tl.uint8)
    sb_odd = tl.minimum(tl.maximum(sb_odd_i32, 0), 255).to(tl.uint8)

    s_ptrs_even = (SCALE_ptr + bh_idx * stride_s_bh
                   + tb_idx * stride_s_tb
                   + d_even * stride_s_d)
    s_ptrs_odd = (SCALE_ptr + bh_idx * stride_s_bh
                  + tb_idx * stride_s_tb
                  + d_odd * stride_s_d)
    tl.store(s_ptrs_even, sb_even, mask=dp_mask)
    tl.store(s_ptrs_odd, sb_odd, mask=dp_mask)

    # ---- Normalize ----
    vn_even = v_even / scale_even[None, :]
    vn_even = tl.minimum(tl.maximum(vn_even, -FP4_MAX), FP4_MAX)
    vn_odd = v_odd / scale_odd[None, :]
    vn_odd = tl.minimum(tl.maximum(vn_odd, -FP4_MAX), FP4_MAX)

    # ---- E2M1 threshold rounding ----
    ax_e = tl.abs(vn_even)
    idx_e = (ax_e > 0.25).to(tl.uint8)
    idx_e += (ax_e > 0.75).to(tl.uint8)
    idx_e += (ax_e > 1.25).to(tl.uint8)
    idx_e += (ax_e > 1.75).to(tl.uint8)
    idx_e += (ax_e > 2.5).to(tl.uint8)
    idx_e += (ax_e > 3.5).to(tl.uint8)
    idx_e += (ax_e > 5.0).to(tl.uint8)
    sign_e = tl.where(vn_even < 0.0, 8, 0).to(tl.uint8)
    nibble_lo = idx_e | sign_e  # [KBLOCK, BLOCK_D_HALF]

    ax_o = tl.abs(vn_odd)
    idx_o = (ax_o > 0.25).to(tl.uint8)
    idx_o += (ax_o > 0.75).to(tl.uint8)
    idx_o += (ax_o > 1.25).to(tl.uint8)
    idx_o += (ax_o > 1.75).to(tl.uint8)
    idx_o += (ax_o > 2.5).to(tl.uint8)
    idx_o += (ax_o > 3.5).to(tl.uint8)
    idx_o += (ax_o > 5.0).to(tl.uint8)
    sign_o = tl.where(vn_odd < 0.0, 8, 0).to(tl.uint8)
    nibble_hi = idx_o | sign_o  # [KBLOCK, BLOCK_D_HALF]

    # ---- Pack: hi << 4 | lo ----
    packed = (nibble_hi << 4) | nibble_lo  # [KBLOCK, BLOCK_D_HALF]

    # ---- Store packed output ----
    p_ptrs = (PACKED_ptr + bh_idx * stride_p_bh
              + t_offs[:, None] * stride_p_t
              + dp_offs[None, :] * stride_p_dhalf)
    tl.store(p_ptrs, packed, mask=dp_mask[None, :])
    

def quantize_fp4_e8m0_per_channel_kblock_triton(
    v: torch.Tensor,
    kblock_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    High-performance FP4 E2M1 quantization with E8M0 per-channel-per-kblock scale.

    Fuses the entire pipeline (amax → exp → scale → normalize → round → pack)
    into a single Triton kernel.

    Args:
        v: Input tensor of shape (BH, T, D). T must be divisible by kblock_size.
           D must be even.
        kblock_size: Token block size for grouping (default 32).

    Returns:
        packed: (BH, T, D//2) uint8 — two E2M1 nibbles per byte
        scale_byte: (BH, T//kblock_size, D) uint8 — E8M0 biased exponent
    """
    assert v.ndim == 3, f"Expected 3D tensor, got {v.ndim}D"
    BH, T, D = v.shape
    assert T % kblock_size == 0, f"T={T} not divisible by kblock_size={kblock_size}"
    assert D % 2 == 0, f"D={D} must be even"
    assert v.is_cuda, "Input must be on CUDA"

    v = v.contiguous()
    num_tblocks = T // kblock_size

    # Allocate outputs
    packed = torch.empty(BH, T, D // 2, dtype=torch.uint8, device=v.device)
    scale_byte = torch.empty(BH, num_tblocks, D, dtype=torch.uint8, device=v.device)

    BLOCK_D_HALF = min(D // 2, 64)

    grid = (BH * num_tblocks, triton.cdiv(D // 2, BLOCK_D_HALF))

    _fp4_perchannel_kblock_kernel_v2[grid](
        v, packed, scale_byte,
        v.stride(0), v.stride(1), v.stride(2),
        packed.stride(0), packed.stride(1), packed.stride(2),
        scale_byte.stride(0), scale_byte.stride(1), scale_byte.stride(2),
        T=T, D=D, KBLOCK=kblock_size,
        BLOCK_D_HALF=BLOCK_D_HALF,
    )

    return packed, scale_byte


# ------------fp8的quant-----可以传入默认粒度=32或者128----
def quantize_fp8_e8m0_triton(
    x: torch.Tensor,
    group_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    High-performance FP8 E4M3 quantization with E8M0 scale.

    Args:
        x: Input tensor, arbitrary shape. Last dim must be divisible by group_size.
        group_size: Quantization group size (default 32).

    Returns:
        x_fp8: Quantized tensor in float8_e4m3fn, same shape as x.
        scale_byte: E8M0 scale as uint8, shape = x.shape[:-1] + (x.shape[-1] // group_size,)
    """
    assert x.shape[-1] % group_size == 0, \
        f"Last dim ({x.shape[-1]}) must be divisible by group_size ({group_size})"
    assert x.is_cuda, "Input must be on CUDA"

    # Flatten to 2D: [num_rows, N]
    orig_shape = x.shape
    N = orig_shape[-1]
    num_rows = x.numel() // N
    x_flat = x.reshape(num_rows, N).contiguous()

    num_groups = N // group_size

    # Allocate outputs
    out_fp8 = torch.empty(num_rows, N, dtype=torch.float8_e4m3fn, device=x.device)
    out_scale = torch.empty(num_rows, num_groups, dtype=torch.uint8, device=x.device)

    if num_groups == 1 or group_size == 32:
        grid = lambda META: (triton.cdiv(num_rows, META["BLOCK_ROWS"]),)
        _quantize_fp8_e8m0_rowblock_kernel[grid](
            x_flat, out_fp8, out_scale,
            N,  # stride_row (contiguous)
            NUM_ROWS=num_rows,
            N=N,
            GROUP_SIZE=group_size,
            NUM_GROUPS=num_groups,
            FP8_MAX=_FP8_MAX,
        )
    else:
        # Choose BLOCK_GROUPS: tile multiple groups per iteration for better utilization
        # BLOCK_GROUPS processes multiple scale groups per row in one program.
        BLOCK_GROUPS = min(num_groups, 16)  # cap at 16 to limit register pressure

        grid = (num_rows,)

        _quantize_fp8_e8m0_vec_kernel[grid](
            x_flat, out_fp8, out_scale,
            N,  # stride_row (contiguous)
            N=N,
            GROUP_SIZE=group_size,
            NUM_GROUPS=num_groups,
            FP8_MAX=_FP8_MAX,
            BLOCK_GROUPS=BLOCK_GROUPS,
        )

    # Reshape outputs to match input shape
    out_fp8 = out_fp8.reshape(orig_shape)
    scale_shape = orig_shape[:-1] + (num_groups,)
    out_scale = out_scale.reshape(scale_shape)

    return out_fp8, out_scale
