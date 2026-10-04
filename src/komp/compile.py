"""Compile KOMP kernels to TTGIR / PTX / cubin for an arch without a GPU, and inspect the result.

Uses the toolchain bundled with the triton wheel (ptxas, cuobjdump). Register / spill counts come from
`ptxas -v`; the SASS instruction mix is counted with KURN's parser (kurn.gpu.sass), so KOMP and KURN-CUDA
kernels are compared the same way.
"""

import os
import re
import subprocess
import tempfile

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from . import kernels

# pointer and K arguments are 16-byte aligned / divisible by 16, as at launch with torch tensors and K % 32 == 0
_GEMV = ({"WQ": "*u8", "WS": "*i16", "XQ": "*i8", "XD": "*fp32", "Y": "*fp32", "S": "*i32", "N": "i32", "K": "i32", "SW": "i32"},
         (0, 1, 2, 3, 4, 5, 7, 8), ("BLOCK_N", "NB", "MODE", "STORE_S"))  # fmt: skip
SIGNATURES = {
    "q4_0_gemv": _GEMV,
    "tq2_0_gemv": _GEMV,
    "quant_q8_0": ({"X": "*fp32", "XQ": "*i8", "XD": "*fp32", "K": "i32"}, (0, 1, 2, 3), ("BB",)),
    "quant_q8_K": ({"X": "*fp32", "XQ": "*i8", "XD": "*fp32", "K": "i32"}, (0, 1, 2, 3), ()),
}
BLOCK = {"q4_0": 32, "tq2_0": 256}
# approximate float ops would make GPU bytes differ from the interpreter's; KOMP kernels must not contain them
APPROX = re.compile(r"\b(div\.full|div\.approx|rcp\.approx|sqrt\.approx|ex2\.approx|lg2\.approx)")


def _bin(name):
    return os.path.join(os.path.dirname(triton.__file__), "backends", "nvidia", "bin", name)


def compile_kernel(name, arch="sm_80", num_warps=4, num_stages=3, **constexprs):
    """-> triton CompiledKernel (asm: ttir, ttgir, llir, ptx, cubin) for `arch` (sm_80, sm_90, ...)."""
    sig, aligned, cnames = SIGNATURES[name]
    fn = getattr(kernels, name)
    if not hasattr(fn, "arg_names"):
        raise RuntimeError("compile needs real JIT functions: unset TRITON_INTERPRET")
    missing = [c for c in cnames if c not in constexprs]
    if missing:
        raise ValueError(f"{name}: missing constexprs {missing}")
    sig = {**sig, **{c: "constexpr" for c in cnames}}
    attrs = {(i,): [["tt.divisibility", 16]] for i in aligned}
    src = ASTSource(fn=fn, signature=sig, constexprs=dict(constexprs), attrs=attrs)
    cc = int(arch.split("_")[1])
    return triton.compile(src, target=GPUTarget("cuda", cc, 32), options={"num_warps": num_warps, "num_stages": num_stages})


def ptxas_info(ck, arch):
    """Registers, spills, shared memory of a compiled kernel (ptxas -v on its PTX)."""
    from kurn.gpu.toolchain import parse_ptxas

    m = re.search(r"\.target\s+(\w+)", ck.asm["ptx"])
    arch = m.group(1) if m else arch  # sm_90 kernels target sm_90a
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "k.ptx")
        with open(p, "w") as fh:
            fh.write(ck.asm["ptx"])
        r = subprocess.run([_bin("ptxas"), "-v", f"--gpu-name={arch}", p, "-o", os.path.join(d, "k.cubin")], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr[-2000:])
    rep = parse_ptxas(r.stderr)
    return next(iter(rep.values()))


def sass(ck):
    """KURN's SASS summary (kurn.gpu.sass.parse) of the kernel's cubin: {"counts", "total", "loop"}."""
    from kurn.gpu import sass as ksass

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "k.cubin")
        with open(p, "wb") as fh:
            fh.write(ck.asm["cubin"])
        r = subprocess.run([_bin("cuobjdump"), "-sass", p], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr[-2000:])
    return next(iter(ksass.parse(r.stdout).values())), r.stdout


def approx_ops(ck):
    return sorted(set(APPROX.findall(ck.asm["ptx"])))


def gemv_mix(fmt, arch="sm_80", block_n=32, nb=4, mode=2, num_warps=4):
    """Compile one GEMV and summarize its hot loop: instructions per weight value per thread (one loop iteration
    covers block_n * nb blocks; Triton does not unroll this loop, which `iters` confirms from the SASS)."""
    ck = compile_kernel(f"{fmt}_gemv", arch, num_warps=num_warps, BLOCK_N=block_n, NB=nb, MODE=mode, STORE_S=False)
    info = ptxas_info(ck, arch)
    rep, _ = sass(ck)
    lp = rep["loop"]["counts"] if rep["loop"] else {}
    per_thread = block_n * nb * BLOCK[fmt] / (num_warps * 32)
    return {"fmt": fmt, "arch": arch, "mode": mode, "block_n": block_n, "nb": nb, "num_warps": num_warps, "regs": info["regs"],
            "spill": info["spill_st"] + info["spill_ld"] + info["stack"], "approx": approx_ops(ck),
            "ptx_dp4a": ck.asm["ptx"].count("dp4a.s32.s32"), "loop_len": rep["loop"]["len"] if rep["loop"] else 0,
            "loop": lp, "values_per_thread": per_thread, "int_per_value": lp.get("int", 0) / per_thread,
            "dp4a_per_value": lp.get("dp4a", 0) / per_thread,
            "insts_per_value": (rep["loop"]["len"] if rep["loop"] else 0) / per_thread}  # fmt: skip


def kurn_cuda_mix(fmt, arch="sm_80", **overrides):
    """The same summary for a KURN-CUDA GEMV (needs nvcc): one hot-loop iteration covers unroll units of unit/sub values."""
    from kurn.gpu import sass as ksass
    from kurn.gpu.spec import FORMATS, resolve

    c = resolve({"op": "gemv", "weights": fmt, "target": "cuda", "arch": arch, **overrides})
    rep = ksass.inspect(c, arch)["kg_gemv"]
    lp = rep["loop"]["counts"] if rep["loop"] else {}
    per_thread = c["unroll"] * FORMATS[fmt]["unit"] / c["sub"]
    return {"fmt": fmt, "arch": arch, "config": f"layout={c['layout']} xlayout={c['xlayout']} tpr={c['tpr']} rpb={c['rpb']} "
            f"sub={c['sub']} unroll={c['unroll']}", "loop_len": rep["loop"]["len"] if rep["loop"] else 0, "loop": lp,
            "values_per_thread": per_thread, "int_per_value": lp.get("int", 0) / per_thread,
            "dp4a_per_value": lp.get("dp4a", 0) / per_thread,
            "insts_per_value": (rep["loop"]["len"] if rep["loop"] else 0) / per_thread}  # fmt: skip
