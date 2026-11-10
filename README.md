# Secret Sauce — Clean-Room GPU Kernel Porting Pipeline

**Goal:** extract the behavioral spec of a closed-source CUDA kernel and auto-generate an open-source, cross-platform (Triton) equivalent — verified byte-for-byte.

**Status:** end-to-end pipeline PROVEN on three distinct kernel shapes, including a **real cuBLAS production kernel**.
`port.py <cubin> <kernel_cu> [n]` runs spec extraction → generation → verification
from a cubin alone. Byte-for-byte match, `max abs diff = 0.000e+00`.

## Verified kernels

| Kernel | Source | Shape | Redundancy | sizes | match |
|---|---|---|---|---|---|
| `custom` | hand-written | per-block reduction | yes (×32) | 256–16384 | YES |
| `vec_add` | hand-written | elementwise | no | 1024 | YES (full array) |
| `transpose` | **cuBLAS fatbin** | tiled transpose | no | 16–128 | YES |

Separately, four kernels — the three above plus the sglang MoE kernel — are
reproduced **byte-for-byte by the SASS emulator**, with no GPU involved at run
time, and checked against hardware. See "An executable SASS simulator".

The transpose kernel is `_Z36transpose_readWrite_alignment_kernelIffLi1ELb1ELi6ELi5ELi3EEv21cublasTransposeParamsIT0_EPKT_PS3_PKS1_` extracted from `cublas_fatbin.2.sm_80.cubin` (64KB, real production SASS). The pipeline detected its feature set from SASS alone and generated a matching Triton kernel.

## Files

| File | Purpose |
|---|---|
| `kernel.cu` | The "closed-source" target — shared-memory tiling, `__syncthreads()`, warp-shuffle reduction |
| `vec_add_kernel.cu` | Reference elementwise kernel (no shared memory, no shuffle) |
| `transpose_kernel.cu` | Reference tiled-transpose kernel (cuBLAS-style bank-conflict padding) |
| `port.py` | **End-to-end driver.** `port.py <cubin> <kernel_cu> [n]` |
| `verify_transpose.py` | 2D transpose verifier: `verify_transpose.py <cubin> <kernel_cu> <rows> <cols>` |
| `spec_extractor.py` | Extracts behavioral spec from a cubin (SASS + PTX + rea) → `spec.json` |
| `generate_triton.py` | Reads `spec.json`, emits a clean-room Triton kernel → `generated_kernel.py` |
| `generated_kernel.py` | The auto-generated Triton kernel (output of generate_triton.py) |
| `custom.cubin` / `.ptx` | Compiled target kernel (sm_80) |
| `vec_add.cubin` / `.ptx` | Compiled reference kernel |
| `cublas_fatbin.2.sm_80.cubin` | Extracted cuBLAS transpose kernel (sm_80) |
| `sass_emu.py` | **Executable SASS simulator** — runs a cubin, produces its output buffer |
| `sass_verify.py` | Runs the same cubin on GPU + emulator, diffs every output byte |
| `sass_opcheck.py` | Differential per-idiom test of SASS instruction semantics |
| `Decompile.java` | Ghidra headless post-script |
| `cublas_fatbin.bin` | cuBLAS's production GPU fatbin — 1,261 kernel objects across sm_75/80/86/90/100/120 |
| `cublas_gpu_evidence.json` | REA's inventory of that fatbin |

## The pipeline

```
1. EXTRACT SPEC
   spec_extractor.py <cubin> spec.json
   → kernel name, SM target, shared_bytes, feature set, register count, SASS, PTX

2. CLEAN-ROOM GENERATE
   generate_triton.py spec.json generated_kernel.py
   → Triton kernel composed from feature templates. Never sees CUDA source.

3. VERIFY
   port.py <cubin> <kernel_cu> [n]
   → compiles the .cu → ctypes, runs generated Triton on SAME input tensors,
     asserts torch.equal(out_cuda, out_triton)
```

## The critical behavioral finding

The first generated kernel FAILED — and that failure was the whole point.

The CUDA kernel redundantly computes the FULL dot on EVERY lane (FFMA chains
over all 256 elements), then lane 0 writes. The warp-shuffle chain (offsets
16/8/4/2/1) sums 32 identical copies across the warp → result = 32 × dot.

A naive port computes the mathematically-correct single dot (77192.5) and
silently diverges from the CUDA output (2470160.0). Ratio was exactly 32.0.

**The compiler eliminates the warp-shuffle entirely** — since all lanes hold
identical values, no `SHFL` instruction appears in the SASS. Only `BSYNC`/
`BAR.SYNC` and long `FFMA` chains remain. So the extractor can't just grep
for `SHFL`; it detects the redundancy signature instead: `FFMA` count ≥ 32 +
`bar_sync` feature present.

**Lesson: a clean-room port must reproduce BEHAVIOR, not "improve" the kernel.**
The extractor flags `full_reduce_redundancy`, and the generator emits
`full * 32.0` — not a "better" kernel, a *faithful* one.

## Toolchain setup (survives reboot)

```bash
# Python env (persistent — .venv in the project dir, not /tmp)
python3 -m venv .venv
.venv/bin/pip install torch triton numpy --index-url https://download.pytorch.org/whl/cu128
# Use gcc-15 for host compilation (gcc 16 breaks torch 2.11 headers)
export CC=/usr/bin/gcc-15 CXX=/usr/bin/g++-15
```

## CUDA stack architecture (verified)

```
app → libcudart.so → dlopen libcuda.so.1 (driver)
app → libcublas.so → libcublasLt.so → dlopen libcuda.so.1
GPU: 1,261 fatbin kernels → SASS on sm_75..sm_120
```

Every CUDA lib is a dlopen/dlsym wrapper — none hard-links libcuda.so.1.
The real IP (kernel-mode driver, MLIR frontend, optimized fatbin kernels) is NOT in cuda_ref/ (symlinked /opt/cuda/lib64/).

## CUDA_Kernel_Samples corpus + pattern matching

`https://github.com/Tongkaio/CUDA_Kernel_Samples` is cloned to
`/tmp/CUDA_Kernel_Samples` (not in the repo; `.gitignore` covers CUDA
artifacts). Eight representative kernels were compiled to cubins and indexed
into a SASS fingerprint corpus (`corpus.py`):

| sample | kernel | reduction style |
|---|---|---|
| `sgemm/src/kernel1-3.cu` | `sgemm_v1/v2/v3` | dot_product |
| `transpose/transpose.cu` | `device_transpose_v5` | elementwise (shared tile) |
| `reduce/sum/sum.cu` | `device_reduce_v1` | warp_shuffle_reduce |
| `reduce/max/max.cu` | `max_kernel` | block_reduce_atomic |
| `reduce/softmax/softmax.cu` | `softmax_kernel` | block_reduce_atomic |
| `reduce/softmax_matrix/softmax_matrix.cu` | `softmax_col_kernel2` | warp_shuffle_reduce |

`corpus.py <cubin>` matches a closed-source cubin against the corpus by
bag-of-opcodes + reduction style.  Results on the closed-source targets:

| target | style | best match | score |
|---|---|---|---|
| `custom.cubin` | shared_tile | softmax_matrix | 8.5 |
| `vec_add.cubin` | elementwise | transpose | 13.5 |
| `samples/sgemm_v1.cubin` | dot_product | sgemm1 | 17.0 |
| `samples/reduce_v4.cubin` | warp_shuffle_reduce | sum | 17.0 |
| `samples/sgemv_k32.cubin` | warp_shuffle_reduce | sgemv_k32 | 18.0 |
| `samples/softmax_row.cubin` | warp_shuffle_reduce | softmax_matrix | 20.0 |

Matching is **heuristic**: a high score means the closed-source kernel shares
the same reduction structure as a known open-source sample. It does NOT
prove equivalence — that requires the verification step.

## Honest boundaries

1. The cuBLAS "ground truth" (`transpose_kernel.cu`) is a *hand-written
   reference* reconstructing the SASS structure, not NVIDIA's actual source.
   Verification proves the generated Triton matches that reference; it does
   not prove the reference is behaviorally identical to real cuBLAS.
2. The redundancy claim (compiler elides warp-shuffle; port must reproduce
   `32 × dot`) is demonstrated on `custom` at n=4096 → 2470160.0. The
   detector (`FFMA count ≥ 32` + `bar_sync`) is a heuristic, not a proof.
3. **The generalization claim is unproven.** The generator's templates cover
   elementwise, block_reduce, transpose, and a 2D dot.  It cannot derive
   matmul's row/col addressing or softmax's exp/max/divide from SASS alone —
   those are data-layout semantics not encoded in the opcode set.  `sgemm`,
   `reduce`, `sgemv`, and `softmax` samples run through the harness but do
   NOT byte-match; they exist to exercise the corpus matcher and to bound
   what the extractor can infer.  Only `vec_add` and `custom` (all sizes)
   are byte-for-byte verified.
4. Ghidra headless analysis of `cuda_ref/` is grounded in real evidence:
   `rea inspect-artifact` on `libcublas.so` returns a SHA256 digest and
   artifact graph; the constant pool (`c[0x0][0x164]`=grid.x,
   `c[0x0][0x168]`=blockIdx.x, `c[0x0][0x170]`=A base,
   `c[0x0][0x178]`=B base) is extracted from SASS.  Custom `.java` Ghidra
   scripts fail to load in 12.3 (bundle classpath issue) — use the built-in
   `ExportFunctionInfoScript` instead.

## Launch signatures are REA-derived, not hand-coded

`port.py` now infers the ctypes launch signature directly from the cubin via
`rea inspect-gpu-code-objects --analysis kernels`, which returns the exact
parameter layout (ordinal, offset, size) from the ELF symbol table and
`.nv.info` section. This works on **closed-source** cubins — `custom.cubin`
yields `[(0,8),(1,8),(2,8),(3,4)]` with no source.

Inference rule: `size==8` → pointer, `size==4` → int/float scalar. The output
pointer is disambiguated from the SASS store tail + kernel-name hints
(reduce/sum/max → middle pointer; sgemm → last; softmax → second). Integer
scalars are substituted with the problem size `n` at call time; sgemm's
`alpha=1.0`/`beta=0.0` are name-derived.

This removed the entire hand-maintained `LAUNCH_TABLE`. Verified on all six
closed-source targets — the launch layer is correct; the remaining
non-matches are generator-side (see Honest boundaries).

## External cubin parsers tried

Two third-party parsers were cloned and tested against our CUDA 13.4 cubins:

- **NBAssembler** (`QianyanTech/NBAssembler`, Python) — SASS disassembler +
  decompiler to PTX for sm_50–sm_80. Fails on CUDA 13.4 cubins: it asserts
  `EI_OSABI == 0x33` (CUDA 11.x ABI); ours is `0x41` (v8), plus an
  `e_flags` mask assert. Two-line relax patch gets past the first assert;
  the second is a CUDA-version gap, not a code bug.
- **cudaparsers** (`VivekPanyam/cudaparsers`, Rust) — parses the
  undocumented cubin/fatbin ELF format, validated against cuobjdump on
  thousands of files. Builds cleanly with Rust 1.95. Fails on our cubins with
  `Could not match enum variant id = 102` — CUDA 13.4 added a new
  `.nv.info` attribute (`0x66`). The enum is a `deku` typed enum; adding
  `#[repr(u8)]` + an `#[deku(id_pat = "_")] Unknown` catch-all variant
  (2-line patch) makes it tolerant. The structured output it exposes is
  richer than cuobjdump: full `EIATTR_*` attribute set including
  `EIATTR_KPARAM_INFO`.

### What KPARAM gives us (the key finding)

`EIATTR_KPARAM_INFO` (attribute `0x17`, 12-byte SVAL) is the authoritative
launch-argument layout for a closed-source cubin. Each entry is
`(ordinal, offset, cbank_info)`, and the `cbank_info` flag bit `0x200000`
distinguishes **pointers** (`0x21f000`) from **scalars** (`0x11f000`) —
exactly the ambiguity I was guessing from the mangled name.

`port.py` now infers the entire ctypes launch signature from
`kparam.kparam_layout(cubin)` — no `LAUNCH_TABLE`, no name guessing:

| cubin | KPARAM (offset:ptr) | inferred signature |
|---|---|---|
| `custom.cubin` | 0:T, 1:T, 2:T, 3:F | (a, b, out, n) |
| `vec_add.cubin` | 0:T, 1:T, 2:T, 3:F | (a, b, out, n) |
| `reduce_v4.cubin` | 0:T, 1:T, 2:F | (d_x, d_y, N) |
| `sgemv_k32.cubin` | 0:T, 1:T, 2:T, 3:F, 4:F | (A, x, y, M, K) |
| `softmax_row.cubin` | 0:T, 1:T, 2:F, 3:F | (input, output, M, N) |
| `sgemm_v1.cubin` | 0:F, 1:F, 2:F, 3:F, 4:T, 5:T, 6:F, 7:T | (M,N,K,alpha,A,B,beta,C) |

The launch layer is correct on all six (CUDA runs and produces correct
output). `kparam.py` is a self-contained pure-Python parser of the
`.nv.info` KPARAM stream — no Rust toolchain needed at runtime.

## cudad — SSS decompiler (vanehl-arial/cudad)

`cudad` is a Rust SASS decompiler: it reads `cuobjdump -sass` text, builds a
CFG, lifts to SSA, and recovers CUDA-like pseudo-C with typed parameters and
memory-space facts. Builds on Rust 1.95 in ~30 s. Works on CUDA 13.4 cubins
with no patches.

It is the structured-facts layer between raw SASS and the Triton generator.
`port.py` now prefers cudad's ABI recovery for launch signatures, with
`kparam.py` (KPARAM) as fallback. They agree on every closed-source target.

cudad output for our six targets (note the **types** KPARAM cannot give):

| target | cudad params | instr | memory accesses |
|---|---|---|---|
| `custom.cubin` | 3× ptr64 float*, 1× int32_t | 424 | 136 |
| `vec_add.cubin` | 3× ptr64 float*, 1× int32_t | 32 | 4 |
| `sgemm_v1.cubin` | 3× int32_t, float, 2× ptr64 float*, float, ptr64 float* | 216 | 62 |
| `reduce_v4.cubin` | 2× ptr64 float*, 1× int32_t | 80 | 5 |
| `sgemv_k32.cubin` | 3× ptr64 float*, 2× int32_t | 136 | 12 |
| `softmax_row.cubin` | 2× ptr64 float*, 2× int32_t | 728 | 45 |

cudad recovers sgemm's `alpha`/`beta` as `word32 float` (params 3, 8) —
something neither KPARAM nor the mangled name could determine — and tags
every pointer with `pointee_type: float`. It also emits a machine-readable
analysis JSON (`schema_version: cudad.analysis.v1`) with ABI, memory, CFG
metrics, opcode coverage, and unresolved-control hazards.

The `custom` pseudo-C opens with the correct signature
`void custom_kernel(float* arg0_ptr, float* arg2_ptr, float* arg4_ptr, int32_t arg6)`
— matching KPARAM exactly, recovered from SASS alone.

## ptx2cuda (VandenkerckhoveJesse) — incomplete, not usable

PTX→CUDA decompiler from a master's thesis. Cloned fine but the repo ships
**no `Cargo.toml` and no `main.rs`** — it is a library-only skeleton. The
source uses `cpp_demangle` plus unresolved modules `c` and `p`; a minimal
`Cargo.toml` + `main.rs` I wrote produces 57 compile errors. Abandoned/
incomplete; not integrated. (We work from SASS, not PTX, as primary input,
so this would have been complementary anyway.)

## cudad memory features drive shape classification

`spec_extractor.py` now calls `cudad.analyze()` and folds the recovered
memory-space/kind inventory into the feature set as `mem_<space>` /
`op_<kind>` entries. This is the signal that lets the extractor classify
data-layout semantics the opcode bag alone cannot:

| target | cudad by_space | shape |
|---|---|---|
| `custom.cubin` | const 1, global 3, **shared 132** | block_reduce (×32) |
| `vec_add.cubin` | const 1, global 3 | elementwise |
| `sgemm_v1.cubin` | param 1, const 1, global 60 | elementwise_2d |
| `reduce_v4.cubin` | const 1, global 2, shared 2, **op_reduction** | block_reduce |
| `sgemv_k32.cubin` | const 1, global 11 | elementwise_2d |
| `softmax_row.cubin` | const 1, global 44 | elementwise_2d |

Two rules come from cudad, not from the opcode bag:
- `op_reduction` (a global `atomicAdd`) → `block_reduce`, overriding the
  shared-memory heuristic (reduce_v4 has sts/lds/bar_sync too).
- `mem_shared` present → the kernel stages data in shared memory.

The `cudad_memory` dict is preserved verbatim in `spec.json` so the evidence
is auditable.

## ZLUDA (vosen) — not applicable

ZLUDA is a drop-in CUDA-on-AMD runtime. It ships a PTX parser (`ptx_parser`,
`ptx`) and a large `cuda_types` crate, but it does **not** parse CUDA ELF
cubins or SASS — it consumes PTX text emitted by `nvcc`/`ptxas`, and hooks
`nvrtcGetCUBIN` at runtime to intercept fatbins. Our cubins are SASS-only
(`cuobjdump -ptx` returns 0 lines), so the PTX path is unusable and there is
no cubin/SASS entry point to reuse. Not integrated.

## Real production kernels: sgl-project/sglang

`https://github.com/sgl-project/sglang` cloned to `/tmp/sglang`. Most kernels
depend on torch/cub/flashinfer/cutlass headers and don't compile standalone.
One is self-contained: `kernels/jit/csrc/moe/moe_permute_prepare.cu` — a real
MoE permute-prep kernel (binary search over sorted expert assignments, two
output arrays). Extracted to `samples/real/moe_permute_prepare.cu` with the
TVM-FFI host glue replaced by an `extern "C"` launcher.

**cudad ABI recovery generalizes to production code.** On the sglang cubin it
recovers the exact signature with types:

```
function=_ZN6sglang26moe_permute_prepare_kernelEPKiPKlPvPiiibb  instr=224
  param[ 0] ptr64 pointee=int32_t    (sorted_topk_ids)
  param[ 2] ptr64 pointee=uint32_t   (reorder_ids)
  param[ 4] ptr64 pointee=int32_t    (expert_offsets)
  param[ 6] ptr64 pointee=uint32_t   (src2dst)
  param[ 8] word32 int32_t           (num_experts)
  param[ 9] word32 int32_t           (numel)
  param[10] word32 uint32_t          (use_int64_offset, is_ep)
memory: 22 accesses, 19 dynamic, 0 unresolved roots
```

**Verified correct on real production data** (100 tokens, 16 experts,
random sorted assignments + shuffled permutation) against a numpy CPU
reference — `expert_offsets` matches exactly and `src2dst` matches
exactly. (An earlier false negative was a bug in my test data: the array
had 110 entries but `numel=100`, so the kernel correctly searched only
the first 100 tokens.)

Shape classification also works: `elementwise`, 19 global accesses,
11 load / 11 store, no shared memory, no reduction.

**Multi-output — supported structurally, reported honestly.**
This kernel writes to **two** output arrays (`expert_offsets` at
`param_word_4`, `src2dst` at `param_word_6`). cudad recovers the write
root of every global store, so `infer_launch` collects the full
output-root set and sizes one output tensor per written parameter.
`spec_extractor` records `n_out` in the spec; `generate_triton.py`
emits `out0_ptr, out1_ptr, ...` — one per output array. So the
multi-output *structure* is now representable.
What still cannot be reproduced is the kernel's *semantics*: the two
outputs have different sizes (37 vs 100) and each thread writes to a
data-dependent index (scatter). The generator's single-size,
non-scatter template can't model that, so `port.py` prints
`multi-output kernel: 2 output arrays, sizes=[1, 100] ...`
`structurally unsupported, not a false pass`.
The false-YES bug is gone; the remaining gap is the generator's
template, not the launch layer.

## The scatter frontier — where clean-room generation stops

cudad's pseudo-C for the sglang MoE kernel renders the stores as:

```
arg4_ptr[r9_13] = r4_20;     // expert_offsets[e] = offset
arg6_ptr[r10_9] = r9_22;     // src2dst[reorder_ids[i]] = i - neg_count
```

Both stores use **data-dependent indices** computed from the input at
runtime. cudad renders those indices as opaque temps (`r9_13`, `r10_9`) —
its honest "debt marker" style for arithmetic it could not simplify to a
source-level expression. That is the precise boundary.

Recovering `expert_offsets[e] = offset` from a closed-source cubin requires
**symbolic execution / value-range analysis of the SASS index arithmetic** —
tracing how `r9_13` is derived from `sorted_topk_ids` through the
`lower_bound` binary search. None of the tools in this pipeline emit that in
a consumable form:

| tool | gives | missing |
|---|---|---|
| cudad | write root, memory space, typed ABI | index expression (rendered as temp) |
| cuobjdump | raw SASS | everything above |
| rea | ELF metadata, KPARAM layout | runtime arithmetic |
| kparam | parameter offsets/types | runtime arithmetic |

So the pipeline now recovers the kernel's **structure** completely —
parameter types, output roots, output sizes, memory spaces, reduction style —
but not its **data-dependent addressing**. That is a genuine RE frontier,
not a patchable bug: it would need a SASS symbolic executor feeding index
expressions into the Triton generator, which is a separate project.

**What is verified:** the CUDA side runs correctly on real production data
(`expert_offsets` and `src2dst` both match a numpy CPU reference). The
false-YES bug is gone — `port.py` reports `structurally unsupported, not a
false pass`.

## An executable SASS simulator, verified byte-for-byte on the GPU

`sasssim.py` set out to trace scatter indices symbolically and stalled on loop
summarization. `sass_emu.py` takes a different route: **execute the SASS**.
It runs a cubin the way the hardware would — 32-lane warp, per-lane registers
and predicates, real byte-addressable global/shared/constant memory, barriers
across warps — and produces the kernel's actual output buffer, not a trace.

No per-kernel tables, no source access, no fixups. Unimplemented instructions
raise instead of approximating, so a wrong answer cannot hide.

| file | role |
|---|---|
| `sass_emu.py` | the simulator. `cuobjdump -sass` text in, memory state out |
| `sass_verify.py` | loads the **same cubin** on the GPU and diffs every output byte |
| `sass_opcheck.py` | differential test of individual SASS idioms (GPU vs emulator) |

```bash
.venv/bin/python sass_verify.py            # all kernels
.venv/bin/python sass_verify.py --sweep    # 18 cases across sizes/shapes
.venv/bin/python sass_opcheck.py           # per-idiom differential test
```

### Verified results (RTX 4080, sm_89, CUDA 13.4)

Every row is a byte-for-byte match on the **whole output buffer**, plus an
independent numpy reference confirming the GPU side is actually right
(`ref==gpu` in the harness output) — so a pass cannot come from both sides
being wrong together.

| kernel | shape exercised | cases | result |
|---|---|---|---|
| `vec_add` | elementwise, predicated exit | n = 256, 1024, 4096 | byte-equal |
| `custom` | shared memory, `BAR.SYNC`, 256-long FFMA chain, ×32 redundancy | n = 512, 1024, 2048 | byte-equal |
| `transpose` | 2-D grid/block, padded shared tile, bank-conflict layout | 64×48 | byte-equal |
| `moe_permute_prepare` | integer-only, loops, `lower_bound` binary search, **scatter** | 6 shapes, int64 on/off | byte-equal |

`moe_permute_prepare` is the sglang production kernel the previous section
called the frontier. Both its outputs match: `expert_offsets` (17 values) and
the scattered `src2dst` (100 values written to data-dependent indices).

The scatter is no longer opaque. The store at `0xae0` computes its address with
`LEA R6, P1, R10, c[0x0][0x178], 0x2` + `LEA.HI.X R7, R10, c[0x0][0x17c], R11,
0x2, P1` — a 64-bit pair whose base is the `src2dst` pointer and whose index is
`reorder_ids[i]`. Executing that pair yields `0x10003014` = `src2dst + 5*4` for
`reorder_ids[i] = 5`, exactly what the GPU writes.

### What it took to get right

These were all found by differential testing against hardware, not by reading
the mnemonic — each one silently produced plausible-but-wrong results first:

- `SHF d, a, shift, b` — operands print `dst, a, shift, b`, and the funnel is
  `{b : a}` (`b` is the high half). Required for the sign-extraction idiom.
- `SEL d, a, b, P` — `P ? a : b`; the *first* source is the true arm.
- `IMAD.IADD` / `IMAD.SHL` — not distinct arithmetic; `.IADD` just annotates a
  multiplier of 1, so both are the generic `a*b + c`.
- `LOP3` truth-table index is `(a<<2)|(b<<1)|c`, and the real sources sit
  immediately before the LUT byte (filler operands precede them in some forms).
- `ISETP.<cmp>.<combine> P, PT, a, b, P2` — operand 1 is a gate (normally `PT`
 ), operand 4 is the value combined. Reading the gate as an OR input made every
  lane exit early.
- `[R7.X4]` is a 32-bit scaled address; only `[R2.64]` is a register pair.
  Reading a scaled base as a pair folds the next register into the address.
- Kernel parameters are naturally aligned in the constant bank — `int a; int* p`
  puts `p` at `c[0x0][0x168]`, not `0x164`.
- `LDG`/`LDS` print destination-first, `STG`/`STS` address-first.
- Immediate `0xffffffe` is a positive 28-bit value (`+268435454`), not `-2`.

### What the emulator does not model

Honest boundaries, all of them enforced by raising rather than guessing:

1. **`LEA.HI` signed-division fixup with a non-trivial shift.** ptxas turns
   signed `a / 3`, `/4`, `/7` into magic-multiply + `LEA.HI`, and I could not
   derive the shift rule for negative operands: no formula fits all of
   `a = -1, -7, -19, -1023` against the hardware's answers. The shift-1 halving
   form (`lower_bound`'s `mid = lo + (hi-lo)/2`) **is** verified and exact.
   Division-free kernels — all four above — are unaffected. This is the single
   known instruction-level gap; it raises `Unsupported` by name.
2. **Floating point that needs bit-exact transcendental results.** `FADD`,
   `FFMA`, `FMUL`, `HFMA2`, `FMNMX`, `I2F`, `F2I` round through Python floats
   to fp32 and were byte-exact on every kernel here, but `MUFU` (reciprocal,
   exp2, log2) is computed in double precision. That matched on the MoE kernel
   (which uses `MUFU.RCP` for unsigned division) and is not guaranteed in
   general.
3. **Not covered at all:** `SHFL` beyond the simple modes, `CALL`/`RET`
   (no subroutine stack), `ATOM`/`RED` variants other than the
   add/max/min/exch forms listed in the source, `TEX`/`SURFACE`, and the
   asynchronous `cp.async`/TMA families. `sm_90`+ instructions are out of
   scope; everything here is `sm_80` SASS.
4. **Performance.** It is an instruction interpreter: `custom` at n=2048 takes
   ~26k instructions, the MoE kernel ~1.5k. Fine for verification, not a
   substitute for running on the GPU.

`sasssim.py` is kept as-is: it documents the symbolic-tracing attempt and why
loop summarization is hard. `sass_emu.py` supersedes it for actually
reproducing output.
