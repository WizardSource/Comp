"""`target triton` specs: KURN's spec syntax driving KOMP kernels.

    kernel     q4_0_gemv_triton
    op         gemv
    weights    q4_0           # q4_0 | tq2_0
    target     triton
    arch       sm_80          # PTX/cubin target for `gen`
    mode       dp4a           # dp4a: packed-word dot products (inline PTX; interpreter: pure-Triton fallback) | bytes
    block_n    64             # rows per Triton program
    nb         4              # weight blocks (32 values for q4_0, 256 for tq2_0) per K step
    num_warps  4
    tune       block_n=16,32,64 nb=2,4,8 num_warps=2,4,8

Problem keys (n, k, m) give the shape `verify` runs on (the interpreter is slow: keep them small).
"""

import itertools

from kurn.spec import SpecError
from kurn.spec import parse as _parse

KEYS = {
    "op": ("gemv",),
    "weights": ("q4_0", "tq2_0"),
    "arch": ("sm_80", "sm_86", "sm_89", "sm_90"),
    "mode": ("dp4a", "bytes"),
    "block_n": (8, 16, 32, 64, 128),
    "nb": (1, 2, 4, 8, 16),
    "num_warps": (1, 2, 4, 8),
}
DEFAULTS = {"arch": "sm_80", "mode": "dp4a", "block_n": 64, "num_warps": 4}
FORMAT_DEFAULTS = {"q4_0": {"nb": 4}, "tq2_0": {"nb": 1}}
BLOCK = {"q4_0": 32, "tq2_0": 256}
PROBLEM = {"q4_0": {"n": 64, "k": 512, "m": 1}, "tq2_0": {"n": 64, "k": 1024, "m": 1}}
COMMON = ("kernel", "target")
MODE = {"bytes": 0, "dp4a": 2}  # compiled; the interpreter runs dp4a as 1 (fallback)


def resolve(spec, overrides=None):
    c = {**spec, **(overrides or {})}
    if c.get("target") != "triton":
        raise SpecError(f"target {c.get('target')!r}: komp handles target triton")
    for k in ("op", "weights"):
        if k not in c:
            raise SpecError(f"missing required key {k!r}")
    if c["weights"] not in KEYS["weights"]:
        raise SpecError(f"weights={c['weights']!r} not allowed for target triton: expected one of {list(KEYS['weights'])}")
    for k in c:
        if k not in KEYS and k not in COMMON and k not in ("n", "k", "m"):
            raise SpecError(f"unknown key {k!r} for target triton: expected one of {sorted((*KEYS, *COMMON, 'n', 'k', 'm'))}")
    for k, allowed in KEYS.items():
        c.setdefault(k, FORMAT_DEFAULTS[c["weights"]].get(k, DEFAULTS.get(k)))
        if c[k] not in allowed:
            raise SpecError(f"{k}={c[k]!r} not allowed for target triton: expected one of {list(allowed)}")
    for k, v in PROBLEM[c["weights"]].items():
        c.setdefault(k, v)
    if c["k"] % BLOCK[c["weights"]]:
        raise SpecError(f"k must be a multiple of {BLOCK[c['weights']]} ({c['weights']} block)")
    c.setdefault("kernel", f"{c['weights']}_{c['op']}_triton")
    return c


def load(path):
    with open(path) as fh:
        return _parse(fh.read())


def configs(spec, tune):
    """Every legal config of the spec's tune space (the spec itself if there is none)."""
    names = list(tune)
    for combo in itertools.product(*(tune[k] for k in names)) if names else [()]:
        try:
            yield resolve(spec, dict(zip(names, combo)))
        except SpecError:
            continue


def label(c):
    return f"mode={c['mode']} block_n={c['block_n']} nb={c['nb']} num_warps={c['num_warps']}"
