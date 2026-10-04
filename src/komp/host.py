"""Host side: KURN split layouts from ggml blocks, launches (CPU interpreter or CUDA)."""

import numpy as np
import torch
import triton

from . import kernels

FORMATS = {  # weights: block values, block bytes, scale offset, activation format
    "q4_0": dict(block=32, nbytes=18, qoff=2, qbytes=16, soff=0, act="q8_0"),
    "tq2_0": dict(block=256, nbytes=66, qoff=0, qbytes=64, soff=64, act="q8_K"),
}
MODES = {"bytes": 0, "dp4a": 2}  # compiled kernels; the interpreter runs dp4a as MODE 1 (the pure-Triton fallback)


def split(fmt, W, n, k):
    """ggml blocks (bytes, n rows) -> (WQ uint8 [n][quant bytes per row], WS int16 [n][blocks]): KURN's split planes."""
    f = FORMATS[fmt]
    blocks = np.frombuffer(W, dtype=np.uint8).reshape(n, k // f["block"], f["nbytes"])
    wq = np.ascontiguousarray(blocks[:, :, f["qoff"] : f["qoff"] + f["qbytes"]]).reshape(n, -1)
    ws = np.ascontiguousarray(blocks[:, :, f["soff"] : f["soff"] + 2]).view(np.int16).reshape(n, k // f["block"])
    return wq, ws


def split_q4_0(W, n, k):
    return split("q4_0", W, n, k)


def tq2_0_qplane(W, nblk):
    """ggml tq2_0 blocks -> the quant plane [nblk][64] (KURN split layout; the fp16 d goes to the scale plane)."""
    return np.ascontiguousarray(np.frombuffer(W, dtype=np.uint8).reshape(nblk, 66)[:, :64])


def q8_0_blocks(xq, xd):
    """q8_0 split planes (numpy) -> ggml q8_0 blocks (bytes): fp16 d then 32 int8 per block."""
    m, k = xq.shape
    out = np.empty((m, k // 32, 34), dtype=np.uint8)
    out[:, :, :2] = xd.astype(np.float16).view(np.uint8).reshape(m, k // 32, 2)
    out[:, :, 2:] = xq.view(np.uint8).reshape(m, k // 32, 32)
    return out.tobytes()


def q8_K_blocks(xq, xd):
    """q8_K split planes (numpy) -> ggml q8_K blocks (bytes): f32 d, 256 int8, 16 int16 bsums."""
    m, k = xq.shape
    nb = k // 256
    out = np.empty((m, nb, 292), dtype=np.uint8)
    out[:, :, :4] = xd.astype(np.float32).view(np.uint8).reshape(m, nb, 4)
    out[:, :, 4:260] = xq.view(np.uint8).reshape(m, nb, 256)
    bs = xq.astype(np.int32).reshape(m, nb, 16, 16).sum(-1).astype(np.int16)
    out[:, :, 260:] = bs.view(np.uint8).reshape(m, nb, 32)
    return out.tobytes()


def act_blocks(fmt, xq, xd):
    return (q8_0_blocks if FORMATS[fmt]["act"] == "q8_0" else q8_K_blocks)(xq, xd)


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def quantize(fmt, x, bb=4):
    """f32 [M][K] tensor -> (XQ int8 [M][K], XD f32 [M][K/block]) with KOMP's quant_q8_0 / quant_q8_K kernel."""
    m, k = x.shape
    xq = torch.empty((m, k), dtype=torch.int8, device=x.device)
    if FORMATS[fmt]["act"] == "q8_0":
        xd = torch.empty((m, k // 32), dtype=torch.float32, device=x.device)
        kernels.quant_q8_0[(triton.cdiv(k // 32, bb), m)](x, xq, xd, k, BB=bb)
    else:
        xd = torch.empty((m, k // 256), dtype=torch.float32, device=x.device)
        kernels.quant_q8_K[(k // 256, m)](x, xq, xd, k)
    return xq, xd


def gemv(fmt, wq, ws, xq, xd, n, k, block_n=32, nb=4, mode=0, num_warps=4, store_s=False):
    """-> (Y f32 [M][N], S int32 [M][N][blocks] or None). mode: 0 bytes, 1 dp4a fallback, 2 dp4a inline PTX."""
    m = xq.shape[0]
    y = torch.empty((m, n), dtype=torch.float32, device=xq.device)
    s = torch.zeros((m, n, k // FORMATS[fmt]["block"]), dtype=torch.int32, device=xq.device) if store_s else y
    fn = kernels.q4_0_gemv if fmt == "q4_0" else kernels.tq2_0_gemv
    fn[(triton.cdiv(n, block_n), m)](
        wq, ws, xq, xd, y, s, n, k, wq.stride(0), BLOCK_N=block_n, NB=nb, MODE=mode, STORE_S=store_s, num_warps=num_warps
    )
    return y, (s if store_s else None)
