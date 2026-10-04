"""Verification of KOMP kernels against KURN's exact references, in the Triton interpreter (TRITON_INTERPRET=1).

Checks, per problem (random and extreme data; row, column and K-step tails):
  dp4a    the pure-Triton dp4a fallback == the PTX dp4a.s32.s32 definition, on edge and random words
  quant   activation blocks from KOMP's quantizers == ggml's reference quantizers (kurn.gpu.ref), byte for byte
  decode  Q4_0 and TQ2_0 decode helpers == KURN's reference values (kurn.formats), every value
  idots   the GEMV's int32 block dots == the exact integer dots of KURN's reference values, bit for bit; and
          kurn.formats.reference_dot rebuilt from them (same float ops, same order) is bit-identical
  output  f32 Y vs the exact reference: relative error <= TOL (KURN's rule; f32 accumulation order differs)

The interpreter cannot execute inline PTX, so the dp4a kernels run here with the fallback (MODE 1). The compiled
MODE 2 kernel is the same program with the PTX instruction in place of the fallback; `komp gen` checks that its PTX
contains dp4a.s32.s32 and its SASS IDP.4A, and the GPU kit checks its results on hardware.
"""

import os
import random
import struct

import numpy as np
import torch

from kurn import formats
from kurn.gpu import ref

from . import host, kernels

TOL = 1e-5
MODE_NAMES = {0: "bytes", 1: "dp4a-fallback", 2: "dp4a-ptx"}


def _need_interpreter():
    if os.environ.get("TRITON_INTERPRET") != "1":
        raise RuntimeError("komp.verify runs in the Triton interpreter: set TRITON_INTERPRET=1 before importing triton")


def dp4a_ref(a, b, c):
    """PTX dp4a.s32.s32: c + sum of the products of the four signed bytes of a and b (int32 wraparound)."""
    sb = lambda v, i: ((v >> (8 * i)) & 0xFF) - (256 if (v >> (8 * i)) & 0x80 else 0)  # noqa: E731
    r = c + sum(sb(a, i) * sb(b, i) for i in range(4))
    return (r + 2**31) % 2**32 - 2**31


def check_dp4a(seed=0, n=1024):
    _need_interpreter()
    rng = random.Random(seed)
    edge = [0, -1, 0x7F7F7F7F, -0x7F7F7F80, 0x01010101, 0x0F0F0F0F, 0x03030303, 2**31 - 1, -(2**31), 0x80000000 - 2**32]
    vals = lambda: [rng.choice(edge) if rng.random() < 0.3 else rng.randrange(-(2**31), 2**31) for _ in range(n)]  # noqa: E731
    a, b = vals(), vals()
    c = [rng.randrange(-(2**20), 2**20) for _ in range(n)]
    out = torch.zeros(n, dtype=torch.int32)
    kernels.dp4a_probe[(n // 256,)](torch.tensor(a, dtype=torch.int32), torch.tensor(b, dtype=torch.int32),
                                    torch.tensor(c, dtype=torch.int32), out, MODE=1)  # fmt: skip
    return out.tolist() == [dp4a_ref(x, y, z) for x, y, z in zip(a, b, c)]


def q4_0_values(W, n, k):
    """KURN's reference values (kurn.formats._vals_q4_0) -> int32 [n][k]."""
    return np.array([formats._vals_q4_0(b) for b in formats.split_blocks("q4_0", W)], dtype=np.int32).reshape(n, k)


def tq2_0_values(W, nblk):
    """Values q - 1 of every tq2_0 block, in KURN's reference order (kurn.formats._ref_tq2_0) -> int32 [nblk][256]."""
    out = np.empty((nblk, 256), dtype=np.int32)
    for b, w in enumerate(formats.split_blocks("tq2_0", W)):
        out[b] = [((w[(v // 128) * 32 + v % 32] >> (2 * ((v % 128) // 32))) & 3) - 1 for v in range(256)]
    return out


def check_decode(seed=0, extreme=False):
    _need_interpreter()
    rng = random.Random(seed)
    res = {}
    W = ref.weight_blocks("q4_0", rng, 37, extreme)
    wq, _ = host.split_q4_0(W, 37, 32)
    out = torch.zeros((37, 32), dtype=torch.int32)
    kernels.q4_0_decode[(3,)](torch.from_numpy(wq), out, 37)
    res["q4_0"] = bool(np.array_equal(out.numpy(), q4_0_values(W, 37, 32)))
    W = ref.weight_blocks("tq2_0", rng, 5, extreme)
    out = torch.zeros((5, 256), dtype=torch.int32)
    kernels.tq2_0_decode[(5,)](torch.from_numpy(host.tq2_0_qplane(W, 5)), out, 5)
    ok = bool(np.array_equal(out.numpy(), tq2_0_values(W, 5)))
    # the dot through the decoded values, scaled as kurn.formats does, must equal reference_dot exactly
    xb = ref.quant_q8_K(ref.activations(random.Random(seed + 1), 256 * 5, 1, block=256)[0])
    wbl, xbl = formats.split_blocks("tq2_0", W), formats.split_blocks("q8_K", xb)
    total = 0.0
    for b in range(5):
        s = int(sum(int(a) * q for a, q in zip(out[b].tolist(), struct.unpack_from("<256b", xbl[b], 4))))
        total += struct.unpack_from("<f", xbl[b], 0)[0] * struct.unpack_from("<e", wbl[b], 64)[0] * s
    res["tq2_0"] = ok and total == formats.reference_dot("tq2_0", wbl, xbl)
    return res


def exact_idots(fmt, W, xq, n, k):
    """Exact int64 block dots [M][N][blocks] of KURN's reference weight values with the int8 activations."""
    blk = host.FORMATS[fmt]["block"]
    vals = q4_0_values(W, n, k) if fmt == "q4_0" else tq2_0_values(W, n * k // 256).reshape(n, k)
    v = vals.reshape(n, k // blk, blk).astype(np.int64)
    q = np.asarray(xq).reshape(-1, k // blk, blk).astype(np.int64)
    return np.einsum("nbi,mbi->mnb", v, q)


def rebuild_reference(fmt, W, xb, s, n, k, m):
    """kurn.formats.reference_dot for every output, with each block's integer dot taken from `s`: the same float
    operations in the same order as KURN's reference (so equal integer dots give a bit-identical result)."""
    f = host.FORMATS[fmt]
    wbl, xbl = formats.split_blocks(fmt, W), formats.split_blocks(f["act"], xb)
    nb = k // f["block"]
    out = []
    for i in range(m):
        for r in range(n):
            total = 0.0
            for b in range(nb):
                w, x, sv = wbl[r * nb + b], xbl[i * nb + b], int(s[i, r, b])
                if fmt == "q4_0":
                    total += struct.unpack_from("<e", w, 0)[0] * struct.unpack_from("<e", x, 0)[0] * sv
                else:
                    total += struct.unpack_from("<f", x, 0)[0] * struct.unpack_from("<e", w, 64)[0] * sv
            out.append(total)
    return out


def check_gemv(fmt, n, k, m, seed=0, extreme=False, block_n=32, nb=4, mode=0, num_warps=4):
    """One problem through KOMP's quantizer + GEMV (mode 0 bytes or 1 dp4a fallback). Returns a dict of results."""
    _need_interpreter()
    W, x = ref.problem(fmt, n, k, m, seed, extreme)
    wq, ws = host.split(fmt, W, n, k)
    xq, xd = host.quantize(fmt, torch.tensor(x, dtype=torch.float32))
    xb = host.act_blocks(fmt, xq.numpy(), xd.numpy())
    quant_ok = xb == ref.act_blocks(fmt, x)
    y, s = host.gemv(fmt, torch.from_numpy(wq), torch.from_numpy(ws), xq, xd, n, k, block_n, nb, mode, num_warps, store_s=True)
    idots_ok = bool(np.array_equal(s.numpy().astype(np.int64), exact_idots(fmt, W, xq.numpy(), n, k)))
    rf = ref.reference(fmt, W, xb, n, k, m)
    rebuilt_ok = rebuild_reference(fmt, W, xb, s, n, k, m) == list(rf)
    err = ref.relerr(y.numpy().reshape(-1).tolist(), rf)
    return {"fmt": fmt, "n": n, "k": k, "m": m, "quant": quant_ok, "idots": idots_ok, "rebuilt": rebuilt_ok, "relerr": err,
            "ok": quant_ok and idots_ok and rebuilt_ok and err <= TOL}  # fmt: skip


# shapes: row tails (n not a multiple of block_n), K steps not a multiple of nb blocks, several columns
PROBLEMS = {"q4_0": ((64, 512, 1), (37, 416, 3), (5, 96, 2)), "tq2_0": ((64, 1024, 1), (37, 768, 3), (5, 1280, 2))}
CONFIGS = {"q4_0": ((32, 4, 4), (16, 2, 2), (64, 8, 4)), "tq2_0": ((32, 2, 4), (16, 1, 2), (64, 4, 4))}


def line(r, block_n, nb, nw, mode):
    return (f"{'ok  ' if r['ok'] else 'FAIL'} {r['fmt']:5} gemv {MODE_NAMES[mode]:13} block_n={block_n} nb={nb} warps={nw} "
            f"n={r['n']} k={r['k']} m={r['m']} quant={'exact' if r['quant'] else 'MISMATCH'} "
            f"idots={'exact' if r['idots'] else 'MISMATCH'} reference_dot={'bit-identical' if r['rebuilt'] else 'DIFFERS'} "
            f"relerr={r['relerr']:.1e}")  # fmt: skip


def run_all(log=print, formats_=("q4_0", "tq2_0"), modes=(0, 1)):
    fails = 0
    ok = check_dp4a(seed=3)
    fails += not ok
    log(f"{'ok  ' if ok else 'FAIL'} dp4a fallback == PTX dp4a.s32.s32 semantics (1024 edge/random words)")
    for extreme in (False, True):
        for f, ok in check_decode(seed=7, extreme=extreme).items():
            fails += not ok
            log(f"{'ok  ' if ok else 'FAIL'} decode {f:5} {'extreme' if extreme else 'random '} (values == KURN reference)")
    for fmt in formats_:
        for mode in modes:
            for block_n, nb, nw in CONFIGS[fmt]:
                for n, k, m in PROBLEMS[fmt]:
                    for seed, extreme in ((1, False), (2, True)):
                        r = check_gemv(fmt, n, k, m, seed, extreme, block_n, nb, mode, nw)
                        fails += not r["ok"]
                        log(line(r, block_n, nb, nw, mode) + (" extreme" if extreme else ""))
    log(f"{fails} failures")
    return fails
