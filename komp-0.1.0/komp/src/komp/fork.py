"""The fork's KURN module, `triton.language.extra.kurn`, as `ktl`.

If the fork is installed (komp/tools/install_fork.sh), that module is used. Otherwise it is loaded from the
patch series itself: the file the patch adds is written to a cache directory and imported under its fork
name, so a stock triton wheel of the pinned version runs KOMP without anything being installed into it
(the GPU kit does this).
"""

import importlib
import importlib.util
import os
import sys

import triton.language.extra as _extra

NAME = "triton.language.extra.kurn"
PATCHES = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "patches")


def added_file(patch, path="python/triton/language/extra/kurn/__init__.py"):
    """Contents of a file a unified diff adds (new file mode)."""
    out, on = [], False
    for line in open(patch).read().splitlines():
        if line.startswith("+++ "):
            on = line[4:].strip() == f"b/{path}"
            continue
        if on and line.startswith("diff --git"):
            break
        if on and line.startswith("+"):
            out.append(line[1:])
        elif on and line.startswith("\\"):
            continue
    if not out:
        raise RuntimeError(f"{patch} does not add {path}")
    return "\n".join(out) + "\n"


def _load():
    try:
        return importlib.import_module(NAME), "installed"
    except ImportError:
        pass
    patch = os.path.join(os.environ.get("KOMP_PATCHES", PATCHES), "0001-language-extra-kurn.patch")
    src = added_file(patch)
    d = os.path.join(os.environ.get("KOMP_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "komp")), "fork")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "triton_language_extra_kurn.py")
    if not os.path.exists(p) or open(p).read() != src:
        with open(p, "w") as fh:
            fh.write(src)
    spec = importlib.util.spec_from_file_location(NAME, p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[NAME] = mod
    spec.loader.exec_module(mod)
    _extra.kurn = mod
    return mod, f"loaded from {os.path.basename(patch)}"


ktl, SOURCE = _load()
