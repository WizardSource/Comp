# COMP

COMP is a KURN-based fork of Triton. KURN is "a tool that writes the math AI models run on": it defines quantized
weight formats, their exact references, data layouts and an energy-ranked tuner. COMP takes that math to GPUs
through Triton's compiler (MLIR/LLVM), and keeps the fork as small as possible so it can follow upstream.

**Private for now.** Do not push COMP to a public remote or propose it upstream. Publishing needs the owner's
employer check first.

## What COMP is

- A **minimal fork** of upstream Triton: one added Python module, `triton.language.extra.kurn`, with KURN's format
  helpers for Triton kernels.
- The **`COMP` package**: Triton kernels built on that module, verification against KURN's exact references, a
  no-GPU compile and inspection tool, and `target triton` specs so KURN drives COMP kernels.

| kernel | what it does |
|---|---|
| `q4_0_gemv` | decode-GEMV for Q4_0 weights × q8_0 activations, on KURN's `split` layout |
| `tq2_0_gemv` | decode-GEMV for TQ2_0 (ternary, BitNet b1.58) weights × q8_K activations |
| `quant_q8_0`, `quant_q8_K` | f32 activations → q8 blocks, byte-identical to ggml's reference quantizers |

Each GEMV has two modes:
- `dp4a`: dot products on packed 4-byte words, through inline PTX `dp4a.s32.s32`;
- `bytes`: every value is unpacked to an integer first.

## Upstream pin and the exact diff

- **Upstream:** triton-lang/triton **v3.8.0**, commit `c01b6774b1865984607d89d89d3a10833de92037`
  (`patches/PIN`, `patches/VERSION`). In the repository, `third_party/triton` is a shallow git submodule at that
  commit, untouched.
- **Diff:** `patches/series` holds one patch, `0001-language-extra-kurn.patch`. It **adds one file**,
  `python/triton/language/extra/kurn/__init__.py`, and **modifies no upstream file**. Triton imports every
  package under `triton/language/extra/` automatically, so no registration line is needed.
- **What the added module provides:**
  - Q4_0 decode: `q4_0_lo`, `q4_0_hi`, `q4_0_block32`;
  - TQ2_0 decode: `tq2_0_trit`;
  - `f16_bits_to_f32`;
  - `dp4a(a, b, c, ASM)`: inline PTX when `ASM`, otherwise `dp4a_emu`, a pure-Triton equivalent the interpreter can run;
  - exact `roundf`/`nearbyint`: `round_half_away`, `round_half_even`;
  - ggml's reference activation quantizers: `q8_0_quantize`, `q8_K_quantize`.

  Only correctly rounded float operations are used, so compiled kernels and the interpreter produce the same bytes.

Fetching the upstream tree (only needed to apply or edit the patch; running COMP does not need it):

```sh
git submodule update --init --depth 1 COMP/third_party/triton
# or, outside this repository:
git clone --depth 1 --branch v3.8.0 https://github.com/triton-lang/triton COMP/third_party/triton
```

## Install

COMP runs on the stock `triton==3.8.0` wheel. The diff is pure Python, so the compiled part (`libtriton`, LLVM)
stays byte-identical to upstream's and no LLVM build is needed.

```sh
pip install triton==3.8.0 torch numpy
pip install -e kurn -e COMP              # KURN and COMP from this repository
COMP/tools/install_fork.sh python3       # apply the patch series, copy the added file into triton 3.8.0
COMP fork                                # status: pin, patch series, where the module comes from
```

- **Uninstall:** `COMP/tools/install_fork.sh python3 --uninstall` removes the added file again.
- **Without installing anything into triton:** if `triton.language.extra.kurn` is not installed, `COMP` loads it
  straight from the patch file (cached under `~/.cache/COMP`, or `$COMP_CACHE`). The GPU kit does this.
- **Changing the fork:** edit files under `third_party/triton` that are listed in `patches/FILES`, then run
  `tools/make_patch.sh`. `tools/apply_patches.sh` re-applies the series and checks the pin.

## Use

```sh
COMP verify --all                         # the full check set, in the Triton interpreter (TRITON_INTERPRET=1 is set for you)
COMP verify COMP/examples/q4_0_gemv_triton.kurn       # every config in a spec's tune space
COMP gen COMP/examples/tq2_0_gemv_triton.kurn arch=sm_90 -o out/   # PTX, TTGIR, cubin; registers, spills, SASS mix
COMP mix --archs sm_80,sm_90              # hot-loop instructions per weight: COMP vs KURN-CUDA (KURN rows need nvcc)
python -m pytest COMP/tests
```

**From KURN.** `kurn check|gen|verify SPEC` runs COMP when the spec says `target triton`. The routing is one file in
KURN, `kurn/src/kurn/ext/COMP.py`. Example spec:

```
kernel     q4_0_gemv_triton
op         gemv
weights    q4_0           # q4_0 | tq2_0
target     triton
arch       sm_80          # sm_80 | sm_86 | sm_89 | sm_90
mode       dp4a           # dp4a | bytes
block_n    64             # rows per Triton program
nb         4              # weight blocks per K step
num_warps  4
tune       mode=dp4a,bytes block_n=32,64 nb=4,8
```

## What is verified (on a CPU-only machine)

**In the Triton interpreter:** `COMP verify --all`, 77 checks, 0 failures.
- **dp4a fallback:** it equals PTX `dp4a.s32.s32` semantics on 1,024 edge-case and random words.
- **Quantizer bytes:** the q8_0 and q8_K blocks from COMP's quantizers equal ggml's reference quantizers
  (`kurn.gpu.ref`) byte for byte.
- **Decode:** every Q4_0 and TQ2_0 value equals KURN's reference values (`kurn.formats`). A TQ2_0 dot product built
  from the decoded values equals `kurn.formats.reference_dot` exactly.
- **GEMV integer part:** for both formats and both modes, every int32 block dot equals the exact integer dot, bit
  for bit. Rebuilding `kurn.formats.reference_dot` from those integers, with the same float operations in the same
  order, gives KURN's reference bit-identically.
- **GEMV f32 output:** relative error 5e-8 to 2e-7 against the exact reference, under KURN's 1e-5 rule. The f32
  output itself is not bit-identical, because the f32 summation order differs.
- **Coverage:** 3 configurations × 3 shapes per format (row tails, K steps not a multiple of the step, 1–3
  activation columns) × random and extreme data.
- **Limit of the interpreter:** it cannot run inline PTX, so it runs the dp4a kernels through the pure-Triton
  fallback. The compiled kernel is the same program with the PTX instruction in its place.

**Compiled without a GPU, for sm_80 and sm_90:**
- every kernel compiles with no register spills and no approximate float operations;
- the dp4a kernels contain `dp4a.s32.s32` in PTX, and the same count of `IDP.4A` in the SASS hot loop;
- weight loads are 128-bit.

Hot-loop instruction mix per weight on sm_80 (static counts from SASS, using KURN's SASS parser; not speed):

| kernel | instructions | integer | dp4a | weight loads |
|---|---|---|---|---|
| COMP Q4_0, `bytes` (64/4/4 warps) | 5.9 | 5.5 | 0 | 128-bit |
| COMP Q4_0, `dp4a` (64/4/4 warps) | 1.9 | 1.1 | 0.28 | 128-bit |
| KURN-CUDA Q4_0, default | 1.7 | 1.2 | 0.25 | 128-bit |
| COMP TQ2_0, `dp4a` (64/1/4 warps) | 1.3 | 0.8 | 0.28 | 128-bit |
| KURN-CUDA TQ2_0, default | 2.0 | 1.4 | 0.25 | mostly 32-bit |

## Status and limits

- **Not run on a GPU.** No speed, energy or on-hardware correctness numbers yet. Every performance statement above
  is a static instruction count.
- **dp4a:** compiled and inspected, but its results have not been checked on hardware.
- **Scope:** decode-GEMV only (Q4_0, TQ2_0). No batched GEMM, no other KURN formats. Batches above 1 run one
  activation column per program, with no reuse of weights across columns.
- **Pinned to Triton 3.8.0.** Other versions are untested.

## Next steps

1. **A100 run** through KURN's GPU kit (`kurn/contrib/gpu-check`): check the dp4a kernels on hardware, time them
   against KURN-CUDA, ggml-cuda and cuBLAS, and tune the COMP configs.
2. **Batched GEMM** with `tl.dot` and exact per-block scales (f16 and int8 tensor-core variants). This is where
   Triton should help KURN most.
3. **More formats:** Q8_0, Q4_K, IQ4_NL, Q2_0, Q1_0, E8P.
4. **Real compiler patches, only with SASS and GPU evidence:** decode after the shared-memory load, packed loads of
   ggml blocks, scales folded into the MMA epilogue. Each becomes a patch in `patches/`, rebased on every Triton release.
5. **Custom-instruction design study:** ternary/low-bit dot products, codebook lookup, decode-in-load.

## License

- **COMP's own code: MIT** (`LICENSE`), matching KURN and ggml.
- **Triton: MIT.** Keep Triton's copyright and license notices with any copy of the fork or its patch.
- **LLVM** (linked into the triton wheel): Apache-2.0 with LLVM exceptions. Keep its license and NOTICE when
  redistributing the wheel. The exception means kernels compiled by COMP carry no LLVM obligations.
- **`ptxas`, `cuobjdump` and `nvdisasm`** ship inside the triton wheel under NVIDIA's CUDA EULA. COMP uses them but
  does not redistribute them.
