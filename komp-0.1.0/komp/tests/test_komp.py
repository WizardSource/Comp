"""KOMP tests. Each check runs `komp` / `kurn` in a subprocess, because TRITON_INTERPRET must be set (verify) or
unset (gen) before triton is imported."""

import os
import random
import subprocess
import sys

import numpy as np
import pytest

KOMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC = os.path.join(KOMP, "examples", "q4_0_gemv_triton.kurn")


def run(*args):
    env = {k: v for k, v in os.environ.items() if k != "TRITON_INTERPRET"}
    return subprocess.run([sys.executable, "-m", *args], capture_output=True, text=True, env=env, timeout=900)


def test_fork_installed():
    r = run("komp", "fork")
    assert r.returncode == 0, r.stdout + r.stderr


def test_patch_series_only_adds_files():
    for name in open(os.path.join(KOMP, "patches", "series")):
        name = name.strip()
        if not name or name.startswith("#"):
            continue
        text = open(os.path.join(KOMP, "patches", name)).read()
        assert "--- /dev/null" in text
        assert "\n--- a/" not in text, f"{name} modifies an upstream file"


def test_interpreter_verification_against_kurn_references():
    r = run("komp", "verify", "--all")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert r.stdout.strip().endswith("0 failures")
    assert "MISMATCH" not in r.stdout and "DIFFERS" not in r.stdout


def test_spec_tune_space_verifies():
    r = run("komp", "verify", SPEC)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]


@pytest.mark.parametrize("arch", ["sm_80", "sm_90"])
def test_ptx_compile_without_gpu(arch, tmp_path):
    r = run("komp", "gen", SPEC, f"arch={arch}", "-o", str(tmp_path))
    assert r.returncode == 0, r.stdout + r.stderr
    ptx = (tmp_path / f"q4_0_gemv_triton.{arch}.ptx").read_text()
    assert f".target {arch}" in ptx
    assert "div.full" not in (tmp_path / f"quant_q8_0.{arch}.ptx").read_text()


def test_kurn_routes_target_triton():
    r = run("kurn", "check", SPEC)
    assert r.returncode == 0 and "target triton" in r.stdout, r.stdout + r.stderr


def test_bad_spec_rejected(tmp_path):
    p = tmp_path / "bad.kurn"
    p.write_text(open(SPEC).read().replace("block_n    64", "block_n    33"))
    r = run("komp", "check", str(p))
    assert r.returncode == 2 and "block_n" in r.stderr


def test_split_layout_is_kurns():
    """host.split_q4_0 == KURN's split repack (kurn.gpu.codegen._repack_body: 16 quant bytes from offset 2, fp16 d)."""
    from komp import host
    from kurn.gpu import ref

    n, k = 3, 96
    W = ref.weights("q4_0", random.Random(0), n, k)
    wq, ws = host.split_q4_0(W, n, k)
    for r in range(n):
        for b in range(k // 32):
            blk = W[(r * (k // 32) + b) * 18 : (r * (k // 32) + b + 1) * 18]
            assert bytes(wq[r, b * 16 : b * 16 + 16]) == blk[2:18]
            assert ws[r, b].tobytes() == blk[:2]
    assert wq.dtype == np.uint8 and ws.dtype == np.int16
