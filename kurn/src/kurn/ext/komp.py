"""Triton backend (`target triton`, KOMP): `kurn check|gen|verify SPEC` on a spec whose target is triton runs
the komp package (komp/ in this repository: KURN's math on a minimal Triton fork). komp is imported lazily and
is optional; without it, triton specs report how to install it."""

import sys

from .. import hooks


def _spec_command(cmd, argv):
    try:
        from komp.cli import spec_command
    except ImportError as e:
        print(f"kurn {cmd}: target triton needs the komp package (pip install -e komp; {e})", file=sys.stderr)
        return 2
    return spec_command(cmd, argv)


hooks.TARGET_BACKENDS["triton"] = _spec_command
