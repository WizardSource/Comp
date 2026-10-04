"""KOMP Triton kernels, built on the fork's KURN helpers (triton.language.extra.kurn, see komp.fork).

Layouts are KURN's `split` layouts (kurn.gpu, `layout split` / `xlayout split`):
  q4_0 weights   quant plane WQ uint8 [N][K/2] (16 bytes per block of 32) + scale plane WS int16 [N][K/32] (fp16 bits)
  tq2_0 weights  quant plane WQ uint8 [N][K/4] (64 bytes per block of 256, ggml order) + scale plane WS int16 [N][K/256]
  q8_0 acts      int8 plane XQ [M][K] + f32 scale plane XD [M][K/32] (the fp16-rounded d)
  q8_K acts      int8 plane XQ [M][K] + f32 scale plane XD [M][K/256]

GEMV MODE: 0 = byte-wise integer math (every value unpacked to int32), 1 = dp4a on packed words via the pure-Triton
fallback (what the interpreter runs), 2 = dp4a via inline PTX (compiled kernels). Modes 1 and 2 are the same
program except for that one helper, and give identical int32 results.
"""

import triton
import triton.language as tl

from .fork import ktl

ONES = tl.constexpr(0x01010101)


@triton.jit
def quant_q8_0(X, XQ, XD, K, BB: tl.constexpr):
    """f32 activations [M][K] -> q8_0 split planes (ggml quantize_row_q8_0_ref). Grid (cdiv(K/32, BB), M)."""
    m = tl.program_id(1).to(tl.int64)
    nblk = K // 32
    blk = tl.program_id(0) * BB + tl.arange(0, BB)
    bm = blk < nblk
    off = m * K + blk[:, None] * 32 + tl.arange(0, 32)[None, :]
    x = tl.load(X + off, mask=bm[:, None], other=0.0)
    q, d = ktl.q8_0_quantize(x)
    tl.store(XQ + off, q.to(tl.int8), mask=bm[:, None])
    tl.store(XD + m * nblk + blk, d.to(tl.float16).to(tl.float32), mask=bm)


@triton.jit
def quant_q8_K(X, XQ, XD, K):
    """f32 activations [M][K] -> q8_K split planes (ggml quantize_row_q8_K_ref; bsums are not needed). Grid (K/256, M)."""
    m = tl.program_id(1).to(tl.int64)
    blk = tl.program_id(0)
    off = m * K + blk * 256 + tl.arange(0, 256)[None, :]
    x = tl.load(X + off)
    q, d = ktl.q8_K_quantize(x)
    tl.store(XQ + off, q.to(tl.int8))
    tl.store(XD + m * (K // 256) + blk + tl.arange(0, 1), d)


@triton.jit
def q4_0_gemv(WQ, WS, XQ, XD, Y, S, N, K, SW, BLOCK_N: tl.constexpr, NB: tl.constexpr, MODE: tl.constexpr, STORE_S: tl.constexpr):
    """Y[m][n] = sum_k w(n, k) * x(m, k), q4_0 weights x q8_0 activations. Grid (cdiv(N, BLOCK_N), M).

    Each step covers NB blocks of 32 values for BLOCK_N rows: exact int32 block dots, then d_w * d_x * dot in f32.
    STORE_S also writes the int32 block dots to S [M][N][K/32]."""
    m = tl.program_id(1).to(tl.int64)
    rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    rmask = rows < N
    r64 = rows.to(tl.int64)
    nblk = K // 32
    b = tl.arange(0, NB)
    acc = tl.zeros([BLOCK_N, NB], dtype=tl.float32)
    for kb in range(0, nblk, NB):
        blk = kb + b
        bmask = blk < nblk
        wmask = rmask[:, None] & bmask[None, :]
        if MODE == 0:
            i = tl.arange(0, 16)
            qs = tl.load(WQ + r64[:, None, None] * SW + blk[None, :, None] * 16 + i[None, None, :], mask=wmask[:, :, None], other=0)
            xo = m * K + blk[:, None] * 32 + i[None, :]
            xl = tl.load(XQ + xo, mask=bmask[:, None], other=0).to(tl.int32)
            xh = tl.load(XQ + xo + 16, mask=bmask[:, None], other=0).to(tl.int32)
            s = tl.sum(ktl.q4_0_lo(qs) * xl[None, :, :], axis=2) + tl.sum(ktl.q4_0_hi(qs) * xh[None, :, :], axis=2)
        else:
            # packed words: weight word w holds values 4w..4w+3 (low nibbles) and 16+4w.. (high nibbles); dp4a on the
            # unsigned nibbles, then subtract 8 * (sum of the block's activations), which is exact in int32
            w = tl.arange(0, 4)
            wq = tl.load(WQ.to(tl.pointer_type(tl.int32)) + r64[:, None, None] * (SW // 4) + blk[None, :, None] * 4 + w[None, None, :],
                         mask=wmask[:, :, None], other=0)  # fmt: skip
            xo = m * (K // 4) + blk[:, None] * 8 + w[None, :]
            xl = tl.load(XQ.to(tl.pointer_type(tl.int32)) + xo, mask=bmask[:, None], other=0)
            xh = tl.load(XQ.to(tl.pointer_type(tl.int32)) + xo + 4, mask=bmask[:, None], other=0)
            lo = wq & 0x0F0F0F0F
            hi = (wq >> 4) & 0x0F0F0F0F
            xl3 = tl.broadcast_to(xl[None, :, :], lo.shape)
            xh3 = tl.broadcast_to(xh[None, :, :], lo.shape)
            s = tl.sum(ktl.dp4a(hi, xh3, ktl.dp4a(lo, xl3, tl.zeros(lo.shape, tl.int32), MODE == 2), MODE == 2), axis=2)
            ones = tl.full(xl.shape, ONES, tl.int32)
            xsum = tl.sum(ktl.dp4a(ones, xh, ktl.dp4a(ones, xl, tl.zeros(xl.shape, tl.int32), MODE == 2), MODE == 2), axis=1)
            s = s - 8 * xsum[None, :]
        if STORE_S:
            tl.store(S + (m * N + r64[:, None]) * nblk + blk[None, :], s, mask=wmask)
        dw = ktl.f16_bits_to_f32(tl.load(WS + r64[:, None] * nblk + blk[None, :], mask=wmask, other=0))
        dx = tl.load(XD + m * nblk + blk, mask=bmask, other=0.0)
        acc += s.to(tl.float32) * dw * dx[None, :]
    tl.store(Y + m * N + rows, tl.sum(acc, axis=1), mask=rmask)


@triton.jit
def tq2_0_gemv(WQ, WS, XQ, XD, Y, S, N, K, SW, BLOCK_N: tl.constexpr, NB: tl.constexpr, MODE: tl.constexpr, STORE_S: tl.constexpr):
    """Y[m][n] = sum_k w(n, k) * x(m, k), tq2_0 weights x q8_K activations. Grid (cdiv(N, BLOCK_N), M).

    Each step covers NB blocks of 256 values for BLOCK_N rows: exact int32 block dots, then d_x * d_w * dot in f32.
    Quant byte h * 32 + i of a block holds values h * 128 + j * 32 + i at bits 2j. STORE_S writes S [M][N][K/256]."""
    m = tl.program_id(1).to(tl.int64)
    rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    rmask = rows < N
    r64 = rows.to(tl.int64)
    nblk = K // 256
    b = tl.arange(0, NB)
    acc = tl.zeros([BLOCK_N, NB], dtype=tl.float32)
    for kb in range(0, nblk, NB):
        blk = kb + b
        bmask = blk < nblk
        wmask = rmask[:, None] & bmask[None, :]
        s = tl.zeros([BLOCK_N, NB], dtype=tl.int32)
        if MODE == 0:
            c = tl.arange(0, 64)
            qs = tl.load(WQ + r64[:, None, None] * SW + blk[None, :, None] * 64 + c[None, None, :], mask=wmask[:, :, None], other=0)
            xo = m * K + blk[:, None] * 256 + (c // 32 * 128 + c % 32)[None, :]
            for j in tl.static_range(4):
                xj = tl.load(XQ + xo + j * 32, mask=bmask[:, None], other=0).to(tl.int32)
                s += tl.sum(ktl.tq2_0_trit(qs, j) * xj[None, :, :], axis=2)
        else:
            # word w (bytes 4w..4w+3) holds, at bits 2j, values (w // 8) * 128 + j * 32 + 4 * (w % 8) + 0..3; dp4a on the
            # crumbs (0..2), then subtract the sum of the block's 256 activations
            w = tl.arange(0, 16)
            wq = tl.load(WQ.to(tl.pointer_type(tl.int32)) + r64[:, None, None] * (SW // 4) + blk[None, :, None] * 16 + w[None, None, :],
                         mask=wmask[:, :, None], other=0)  # fmt: skip
            xo = m * (K // 4) + blk[:, None] * 64 + (w // 8 * 32 + w % 8)[None, :]
            ones = tl.full([NB, 16], ONES, tl.int32)
            acc3 = tl.zeros(wq.shape, tl.int32)
            xs = tl.zeros([NB, 16], tl.int32)
            for j in tl.static_range(4):
                xj = tl.load(XQ.to(tl.pointer_type(tl.int32)) + xo + j * 8, mask=bmask[:, None], other=0)
                acc3 = ktl.dp4a((wq >> (2 * j)) & 0x03030303, tl.broadcast_to(xj[None, :, :], wq.shape), acc3, MODE == 2)
                xs = ktl.dp4a(ones, xj, xs, MODE == 2)
            s = tl.sum(acc3, axis=2) - tl.sum(xs, axis=1)[None, :]
        if STORE_S:
            tl.store(S + (m * N + r64[:, None]) * nblk + blk[None, :], s, mask=wmask)
        dw = ktl.f16_bits_to_f32(tl.load(WS + r64[:, None] * nblk + blk[None, :], mask=wmask, other=0))
        dx = tl.load(XD + m * nblk + blk, mask=bmask, other=0.0)
        acc += dx[None, :] * dw * s.to(tl.float32)
    tl.store(Y + m * N + rows, tl.sum(acc, axis=1), mask=rmask)


@triton.jit
def dp4a_probe(A, B, C, OUT, MODE: tl.constexpr):
    """Probe: OUT[i] = dp4a(A[i], B[i], C[i]) through ktl.dp4a (MODE 2: inline PTX, else the fallback). Grid (cdiv(n, 256),)."""
    i = tl.program_id(0) * 256 + tl.arange(0, 256)
    tl.store(OUT + i, ktl.dp4a(tl.load(A + i), tl.load(B + i), tl.load(C + i), MODE == 2))


@triton.jit
def q4_0_decode(WQ, OUT, R):
    """Probe: [R][16] q4_0 quant bytes -> [R][32] int32 values via ktl.q4_0_block32. Grid (cdiv(R, 16),)."""
    r = tl.program_id(0) * 16 + tl.arange(0, 16)
    qs = tl.load(WQ + r[:, None] * 16 + tl.arange(0, 16)[None, :], mask=(r < R)[:, None], other=0)
    tl.store(OUT + r[:, None] * 32 + tl.arange(0, 32)[None, :], ktl.q4_0_block32(qs), mask=(r < R)[:, None])


@triton.jit
def tq2_0_decode(WQ, OUT, NBLK):
    """Probe: tq2_0 quant planes [NBLK][64] -> [NBLK][256] int32 values via ktl.tq2_0_trit. Grid (NBLK,)."""
    b = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, 64)
    qs = tl.load(WQ + b * 64 + c)
    v = (c // 32) * 128 + c % 32  # value index of crumb 0 of byte c
    tl.store(OUT + b * 256 + v, ktl.tq2_0_trit(qs, 0))
    tl.store(OUT + b * 256 + v + 32, ktl.tq2_0_trit(qs, 1))
    tl.store(OUT + b * 256 + v + 64, ktl.tq2_0_trit(qs, 2))
    tl.store(OUT + b * 256 + v + 96, ktl.tq2_0_trit(qs, 3))
