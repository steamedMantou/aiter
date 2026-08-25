# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""dsv4_mla_prefill reads the paged KV cache exactly as it was written.

The kernel requantises each staged tile in LDS and takes its softmax frame from
that tile, so no pass has to rewrite the cache to one exponent per token first.
These tests pin that property against a torch reference that decodes the pool
block-64 scale by block-64 scale -- the layout vLLM actually stores.
"""

from __future__ import annotations

import math

import pytest
import torch

from aiter.ops.dsv4_mla_prefill import dsv4_mla_prefill

_D_NOPE, _D_ROPE = 448, 64
_ROW, _NBLK = 576, 7
_DEV = "cuda"


def _skip_if_unsupported() -> bool:
    if not torch.cuda.is_available():
        return True
    return "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName


pytestmark = pytest.mark.skipif(_skip_if_unsupported(), reason="needs gfx950")


class _Pool:
    """A paged pool laid out the way exp_off()/grid_row() address it.

    ``[num_pages, bytes_per_page]`` uint8.  Region A is ``page_size`` rows of
    576 B (448 NoPE fp8 | 128 B RoPE bf16); region B is ``page_size`` slots of
    8 B holding the seven UE8M0 block-64 exponents.
    """

    page_size, num_pages = 64, 96

    def __init__(self, seed: int = 11, hot_row0: bool = False):
        ps, npg = self.page_size, self.num_pages
        self.bytes_per_page = -(-ps * (_ROW + 8) // _ROW) * _ROW
        self.rows_per_page = self.bytes_per_page // _ROW
        self.scale_off = ps * _ROW
        self.page_shift = ps.bit_length() - 1
        self.num_tokens = npg * ps
        g = torch.Generator(device=_DEV).manual_seed(seed)

        real = torch.randn(self.num_tokens, _D_NOPE, device=_DEV, generator=g) * 0.5
        if hot_row0:
            # A partial tile's padding slots read the index array out of bounds,
            # get 0, and gather row 0.  Give row 0 an exponent 20 octaves up --
            # what a recycled block's stale tail looks like.
            real[0] *= 2.0**20
        blk = real.reshape(self.num_tokens, _NBLK, 64)
        # per-block spread, so a token's seven scales are genuinely different
        blk = blk * torch.exp2(
            -torch.randint(0, 2, (self.num_tokens, _NBLK, 1), device=_DEV,
                           generator=g).float()
        )
        e = torch.ceil(
            torch.log2(blk.abs().amax(-1).clamp(min=1e-30) / 448.0)
        ).to(torch.int32)
        self.exp_b = (e + 127).clamp(1, 200).to(torch.uint8)
        q8 = (
            blk / torch.exp2((self.exp_b.int() - 127).float()).unsqueeze(-1)
        ).to(torch.float8_e4m3fn)
        self.rope = (
            torch.randn(self.num_tokens, _D_ROPE, device=_DEV, generator=g) * 0.5
        ).to(torch.bfloat16)

        self.buf = torch.zeros(npg, self.bytes_per_page, dtype=torch.uint8, device=_DEV)
        nope_b = q8.reshape(self.num_tokens, _D_NOPE).view(torch.uint8)
        rope_b = self.rope.view(torch.uint8).reshape(self.num_tokens, 128)
        for p in range(npg):
            base, row = p * ps, self.buf[p]
            for j in range(ps):
                row[j * _ROW: j * _ROW + _D_NOPE] = nope_b[base + j]
                row[j * _ROW + _D_NOPE: (j + 1) * _ROW] = rope_b[base + j]
                s = self.scale_off + j * 8
                row[s: s + _NBLK] = self.exp_b[base + j]

    def views(self):
        """The strided fp8/bf16 views the kernel takes as unified_kv_*."""
        grid = self.num_pages * self.rows_per_page
        flat = self.buf.view(-1)
        nope = torch.as_strided(
            flat.view(torch.float8_e4m3fn), (grid, 512), (_ROW, 1)
        )
        rope = torch.as_strided(
            flat.view(torch.bfloat16), (grid, 64), (_ROW // 2, 1),
            storage_offset=_D_NOPE // 2,
        )
        return nope, rope

    def decode(self) -> torch.Tensor:
        """Dequantise the pool as it stands, one block-64 scale at a time."""
        out = torch.empty(self.num_tokens, _D_NOPE, dtype=torch.float32, device=_DEV)
        for p in range(self.num_pages):
            row = self.buf[p]
            for j in range(self.page_size):
                v = row[j * _ROW: j * _ROW + _D_NOPE].view(torch.float8_e4m3fn)
                s = self.scale_off + j * 8
                sc = row[s: s + _NBLK].int()
                out[p * self.page_size + j] = v.float() * torch.exp2(
                    (sc - 127).float()
                ).repeat_interleave(64)
        return out


def _run(pool: _Pool, n_prefix: int, n_extend: int = 0, seed: int = 7):
    """Call the kernel and the torch reference on the same random query set."""
    n_tok, n_head = 512, 128
    g = torch.Generator(device=_DEV).manual_seed(seed)
    q_nope = (torch.randn(n_tok, n_head, _D_NOPE, device=_DEV, generator=g) * 0.5).to(
        torch.bfloat16
    )
    q_rope = (torch.randn(n_tok, n_head, _D_ROPE, device=_DEV, generator=g) * 0.5).to(
        torch.bfloat16
    )
    sink = torch.randn(n_head, device=_DEV, dtype=torch.float32, generator=g) * 0.25
    scale = 1.0 / math.sqrt(512)

    def csr(width):
        idx = torch.randint(
            0, pool.num_tokens, (n_tok * width,), dtype=torch.int32, device=_DEV
        )
        ptr = torch.arange(
            0, (n_tok + 1) * width, width, dtype=torch.int32, device=_DEV
        )
        return idx, ptr

    ix_p, ip_p = csr(n_prefix)
    if n_extend:
        ix_e, ip_e = csr(n_extend)
    else:
        ix_e = torch.zeros(0, dtype=torch.int32, device=_DEV)
        ip_e = torch.zeros(n_tok + 1, dtype=torch.int32, device=_DEV)

    nope, rope = pool.views()
    max_e = torch.zeros(1, dtype=torch.int32, device=_DEV)
    out = dsv4_mla_prefill(
        q_nope=q_nope, q_rope=q_rope,
        unified_kv_nope=nope, unified_kv_rope=rope,
        kv_indices_prefix=ix_p, kv_indptr_prefix=ip_p,
        kv_nope=nope, kv_rope=rope,
        kv_indices_extend=ix_e, kv_indptr_extend=ip_e,
        attn_sink=sink, kv_max_e=max_e, softmax_scale=scale,
        page_shift_prefix=pool.page_shift,
        rows_per_page_prefix=pool.rows_per_page,
        scale_off_prefix=pool.scale_off,
        page_shift_extend=pool.page_shift,
        rows_per_page_extend=pool.rows_per_page,
        scale_off_extend=pool.scale_off,
    ).float()

    kv = torch.cat([pool.decode(), pool.rope.float()], -1)
    qf = torch.cat([q_nope.float(), q_rope.float()], -1)
    idx = ix_p.reshape(n_tok, n_prefix).long()
    if n_extend:
        idx = torch.cat([idx, ix_e.reshape(n_tok, n_extend).long()], dim=1)
    ref = torch.empty(n_tok, n_head, 512, dtype=torch.float32, device=_DEV)
    for t in range(0, n_tok, 64):        # chunked: the full logit tensor is 8 GB
        sl = slice(t, t + 64)
        k = kv[idx[sl]]
        logits = torch.einsum("bhd,bkd->bhk", qf[sl], k) * scale
        logits = torch.cat(
            [logits, sink.view(1, n_head, 1).expand(logits.shape[0], n_head, 1)], -1
        )
        ref[sl] = torch.einsum(
            "bhk,bkd->bhd", torch.softmax(logits, -1)[..., : idx.shape[1]], k[..., :512]
        )

    a, b = out.reshape(-1, 512), ref.reshape(-1, 512)
    return ((a - b).norm(dim=1) / b.norm(dim=1).clamp(min=1e-30)).median().item()


# fp8 e4m3 on both GEMMs; 6% is the quantisation floor, not a tuned threshold
_TOL = 0.06


def test_reads_cache_as_written():
    """No pass touches the pool between the writes and the kernel."""
    assert _run(_Pool(), n_prefix=512) < _TOL


def test_partial_tile_ignores_padding_rows():
    """A tile's frame must come from its real tokens only.

    With nnz not a multiple of KV_TILE the tail slots gather row 0, and taking
    their exponent into the tile frame drove every real token in that tile to
    underflow -- 0.42 relative error here, and 60 GSM8K points end to end.
    """
    assert _run(_Pool(hot_row0=True), n_prefix=500) < _TOL


def test_extend_segment():
    """Prefix and SWA segments share the accumulator, so both must land."""
    assert _run(_Pool(), n_prefix=500, n_extend=128) < _TOL
