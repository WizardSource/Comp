"""komp command line.

    komp fork                         fork status: pinned upstream, patch series, installed triton
    komp check  SPEC [k=v ...]        validate a `target triton` spec (and its tune space)
    komp gen    SPEC [k=v ...] [-o DIR] [--emit ptx,ttgir,cubin]
                                      compile without a GPU: PTX / TTGIR / cubin for the spec's arch, ptxas
                                      registers and spills, approximate-op lint, dp4a in PTX, hot-loop SASS mix
    komp verify SPEC [k=v ...]        every config of the spec on the Triton interpreter vs KURN's exact references
    komp verify --all                 the full verification set (dp4a fallback, decode helpers, both GEMVs, both modes)
    komp mix [--archs sm_80,sm_90]    hot-loop instruction mix per weight: KOMP (bytes, dp4a) vs KURN-CUDA (needs nvcc)

`verify` runs with TRITON_INTERPRET=1 and `gen`/`mix` without it; komp re-runs itself with the right setting.
`kurn check|gen|verify SPEC` routes here for specs with `target triton` (kurn.ext.komp).
"""

import os
import subprocess
import sys

USAGE = __doc__


def _mode(argv, interpret):
    """Re-run in a subprocess if the interpreter setting is wrong for this command (it is read at triton import)."""
    have = os.environ.get("TRITON_INTERPRET") == "1"
    if have == interpret:
        return None
    env = dict(os.environ)
    if interpret:
        env["TRITON_INTERPRET"] = "1"
    else:
        env.pop("TRITON_INTERPRET", None)
    return subprocess.run([sys.executable, "-m", "komp", *argv], env=env).returncode


def _spec(args):
    from kurn.spec import parse_overrides

    from . import spec

    skip = {args[i + 1] for i, a in enumerate(args[:-1]) if a in ("-o", "--emit")}
    path = next(a for a in args if not a.startswith("-") and "=" not in a and a not in skip)
    base, tune = spec.load(path)
    single, lists = parse_overrides([a for a in args if "=" in a and not a.startswith("-")])
    base.update(single)
    tune.update(lists)
    return spec, base, tune


def _mix_line(r, name):
    lp = r["loop"]
    return (f"{name:34} hot loop {r['loop_len']:4} insts / {r['values_per_thread']:5.0f} weights per thread: "
            f"{r['insts_per_value']:.2f} insts, {r['int_per_value']:.2f} int, {r['dp4a_per_value']:.3f} dp4a per weight; loads "
            f"128-bit {lp.get('ldg128', 0)}, 64-bit {lp.get('ldg64', 0)}, 32-bit {lp.get('ldg32', 0)}, 16-bit {lp.get('ldg16', 0)}, "
            f"8-bit {lp.get('ldg8', 0)}")  # fmt: skip


def cmd_fork(_args):
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    pin = open(os.path.join(root, "patches", "PIN")).read().strip()
    print(f"upstream: triton-lang/triton v{open(os.path.join(root, 'patches', 'VERSION')).read().strip()} ({pin})")
    series = [ln.strip() for ln in open(os.path.join(root, "patches", "series")) if ln.strip() and not ln.startswith("#")]
    print(f"patch series: {', '.join(series)}")
    try:
        import triton
    except ImportError:
        print("triton not installed (pip install triton==3.8.0, then komp/tools/install_fork.sh)")
        return 1
    from . import fork

    print(f"installed triton: {triton.__version__} at {os.path.dirname(triton.__file__)}")
    print(f"triton.language.extra.kurn: {fork.SOURCE}")
    return 0 if fork.SOURCE == "installed" else 1


def cmd_check(args):
    from kurn.spec import SpecError

    spec, base, tune = _spec(args)
    try:
        c = spec.resolve(base)
        n = len(list(spec.configs(base, tune)))
    except SpecError as e:
        print(f"komp check: {e}", file=sys.stderr)
        return 2
    print(f"ok {c['kernel']}: {c['weights']} {c['op']} target triton {c['arch']} {spec.label(c)}; tune space: {n} legal configs")
    return 0


def cmd_gen(args):
    r = _mode(["gen", *args], interpret=False)
    if r is not None:
        return r
    from . import compile as kc

    spec, base, tune = _spec(args)
    c = spec.resolve(base)
    out = args[args.index("-o") + 1] if "-o" in args else None
    emit = (args[args.index("--emit") + 1] if "--emit" in args else "ptx,ttgir,cubin").split(",")
    mode = spec.MODE[c["mode"]]
    fails = 0
    r = kc.gemv_mix(c["weights"], c["arch"], c["block_n"], c["nb"], mode, c["num_warps"])
    bad = r["spill"] or r["approx"] or (mode == 2 and not r["ptx_dp4a"]) or (mode == 2 and not r["loop"].get("dp4a"))
    fails += bool(bad)
    print(f"{'FAIL' if bad else 'ok  '} {c['kernel']} {c['arch']} {spec.label(c)}: regs={r['regs']} spill={r['spill']} "
          f"approx-ops={','.join(r['approx']) or 'none'} PTX dp4a.s32.s32={r['ptx_dp4a']} SASS IDP.4A in hot loop={r['loop'].get('dp4a', 0)}")  # fmt: skip
    print("     " + _mix_line(r, "hot loop"))
    quant = "quant_q8_0" if c["weights"] == "q4_0" else "quant_q8_K"
    qk = kc.compile_kernel(quant, c["arch"], **({"BB": 4} if quant == "quant_q8_0" else {}))
    qi, qa = kc.ptxas_info(qk, c["arch"]), kc.approx_ops(qk)
    qbad = qi["spill_st"] + qi["spill_ld"] + qi["stack"] or qa
    fails += bool(qbad)
    print(f"{'FAIL' if qbad else 'ok  '} {quant} {c['arch']}: regs={qi['regs']} spill={qi['spill_st'] + qi['spill_ld'] + qi['stack']} "
          f"approx-ops={','.join(qa) or 'none'}")  # fmt: skip
    if out:
        os.makedirs(out, exist_ok=True)
        ck = kc.compile_kernel(f"{c['weights']}_gemv", c["arch"], num_warps=c["num_warps"], BLOCK_N=c["block_n"], NB=c["nb"], MODE=mode,
                               STORE_S=False)  # fmt: skip
        for name, k in ((c["kernel"], ck), (quant, qk)):
            for e in emit:
                data = k.asm[e]
                p = os.path.join(out, f"{name}.{c['arch']}.{e}")
                with open(p, "wb" if isinstance(data, bytes) else "w") as fh:
                    fh.write(data)
                print(f"     wrote {p}")
    return 1 if fails else 0


# KOMP configs and KURN-CUDA configs compared by `komp mix`
MIX_KOMP = {"q4_0": ((64, 4, 4), (32, 8, 4)), "tq2_0": ((64, 1, 4), (32, 2, 4))}
MIX_KURN = {"q4_0": ({}, {"sub": 2, "unroll": 2}), "tq2_0": ({},)}


def cmd_mix(args):
    r = _mode(["mix", *args], interpret=False)
    if r is not None:
        return r
    from . import compile as kc

    archs = (args[args.index("--archs") + 1] if "--archs" in args else "sm_80").split(",")
    fails = 0
    for arch in archs:
        print(f"== {arch}")
        for fmt in ("q4_0", "tq2_0"):
            for bn, nb, nw in MIX_KOMP[fmt]:
                for mode in (0, 2):
                    r = kc.gemv_mix(fmt, arch, bn, nb, mode, nw)
                    fails += bool(r["spill"] or r["approx"])
                    name = f"KOMP {fmt} {'dp4a ' if mode else 'bytes'} {bn}/{nb}/{nw}w r{r['regs']}" + (" SPILL" if r["spill"] else "")
                    print(_mix_line(r, name))
            for ov in MIX_KURN[fmt]:
                try:
                    r = kc.kurn_cuda_mix(fmt, arch, **ov)
                except Exception as e:  # noqa: BLE001  (nvcc missing: KOMP rows still print)
                    print(f"KURN-CUDA {fmt}: not available ({str(e).splitlines()[0]})")
                    break
                print(_mix_line(r, f"KURN-CUDA {fmt} {'default' if not ov else 'sub2/unroll2'}"))
    return 1 if fails else 0


def cmd_verify(args):
    r = _mode(["verify", *args], interpret=True)
    if r is not None:
        return r
    from . import verify

    if "--all" in args:
        return 1 if verify.run_all() else 0
    spec, base, tune = _spec(args)
    fails = 0
    ok = verify.check_dp4a()
    fails += not ok
    print(f"{'ok  ' if ok else 'FAIL'} dp4a fallback == PTX dp4a.s32.s32 semantics")
    for c in spec.configs(base, tune):
        mode = 0 if c["mode"] == "bytes" else 1
        for seed, extreme in ((1, False), (2, True)):
            res = verify.check_gemv(c["weights"], c["n"], c["k"], c["m"], seed, extreme, c["block_n"], c["nb"], mode, c["num_warps"])
            fails += not res["ok"]
            print(verify.line(res, c["block_n"], c["nb"], c["num_warps"], mode) + (" extreme" if extreme else "") + "  [interpreter]")
    print(f"{fails} failures")
    return 1 if fails else 0


COMMANDS = {"fork": cmd_fork, "check": cmd_check, "gen": cmd_gen, "verify": cmd_verify, "mix": cmd_mix}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in COMMANDS:
        print(USAGE)
        return 0 if argv and argv[0] in ("-h", "--help") else 2
    return COMMANDS[argv[0]](argv[1:])


def spec_command(cmd, argv):
    """Entry for kurn.hooks.TARGET_BACKENDS["triton"]."""
    if cmd not in ("check", "gen", "verify"):
        print(f"kurn {cmd}: not available for target triton yet (komp supports check, gen, verify)", file=sys.stderr)
        return 2
    return main([cmd, *argv])
