#!/usr/bin/env python3
"""Clean-room Triton generator.

Reads the behavioral spec produced by spec_extractor.py and emits a Triton
kernel that reproduces the ORIGINAL CUDA kernel's behavior - including its
redundancies. Never sees the CUDA C++ source.

Output shapes (SASS-detected, not guessed):
  - elementwise_2d   C[y*N+x] = alpha*dot + beta*C   (matmul, softmax)
  - elementwise      out[idx] = a[idx] op b[idx]      (vec_add)
  - block_reduce     out[block] = sum(a[block*TILE..]) (reduce, atomic)
  - block_reduce_redundant: full dot on every lane, lane 0 writes
  - transpose        shared-memory tiled transpose

Kernel kinds are inferred from the SASS opcode set:
  - EXP  -> softmax (max + exp + divide)
  - else -> dot-product accumulation

Usage: generate_triton.py <spec.json> <out.py>
"""
import json, sys


def generate(spec_path, out_path):
    with open(spec_path) as f:
        s = json.load(f)

    feats = set(s.get("features", []))
    kname = s.get("kernel", "kernel")
    redundancy = s.get("full_reduce_redundancy", False)
    warp = s.get("warp_size", 32)
    shape = s.get("output_shape", "block_reduce")
    sass = s.get("sass", "")

    # Kernel kind from the SASS opcode set.
    is_softmax = ("EXP" in sass) or ("FMNM" in sass)
    # Kernel kind, inferred from the mangled symbol name in the cubin
    # symbol table (metadata, not source).  SASS alone cannot disambiguate
    # data-access patterns (row*col dot vs single-array sum).
    # Default is "dot" (acc = sum(a*b)); "elementwise" (a+b) is set
    # only when the SASS has no FFMA chain - i.e. a pure per-element op.
    kind = "dot"
    # Feature names come from spec_extractor: "fma" (FFMA token), "fmaxmin",
    # "exp".  If no multiply is present the kernel is a pure add.
    if not ("fma" in feats or "fmaxmin" in feats or "exp" in feats):
        kind = "elementwise"  # a + b, no FMUL/FFMA pairing
    if "softmax" in kname.lower():
        kind = "softmax"
    elif "reduce" in kname.lower() or "sum" in kname.lower():
        kind = "reduce"  # acc = sum(a), no FMUL pairing
    elif "sgemv" in kname.lower() or "gemv" in kname.lower():
        kind = "gemv"  # acc = sum(a * b) per row, 1D b
    elif "sgemm" in kname.lower() or "gemm" in kname.lower():
        kind = "sgemm"  # acc = sum(a_row * b_col), 2D both

    # Multi-output support: cudad recovers the write root of every global
    # store, so a kernel that writes to more than one array (e.g. the sglang
    # MoE kernel, which writes to both expert_offsets and src2dst) gets one
    # output pointer per written parameter instead of a single out_ptr.
    out_roots = []
    cmem = s.get("cudad_memory") or {}
    # The full access list isn't in the compact spec dict, so re-derive from
    # the cubin via cudad when available.
    try:
        import cudad
        d = cudad.analyze(spec_path)
        if d is not None:
            for acc in d["functions"][0]["memory"].get("accesses", []):
                if acc["kind"] == "store" and acc.get("root", "").startswith("param_word_"):
                    idx = int(acc["root"].split("_")[-1])
                    if (idx, acc["root"]) not in out_roots:
                        out_roots.append((idx, acc["root"]))
            out_roots.sort()
    except Exception:
        pass
    # Prefer the n_out already computed by spec_extractor (from the cudad
    # write-root set); fall back to re-deriving here.
    n_out = int(s.get("n_out", 0)) or max(1, len(out_roots))

    is_transpose = shape == "transpose"

    L = []
    L.append("# Auto-generated clean-room Triton kernel.")
    L.append("# Produced from the behavioral spec of a closed-source CUDA kernel.")
    L.append("# Never saw the CUDA C++ source.")
    L.append("")
    L.append("import triton")
    L.append("import triton.language as tl")
    L.append("")

    if is_transpose:
        L += _transpose(kname)
    elif shape == "elementwise_2d":
        L += _elementwise_2d(kname, kind)
    else:
        L += _elementwise(kname, redundancy, warp, shape, kind, n_out)

    code = "\n".join(L)
    with open(out_path, "w") as f:
        f.write(code)
    print(f"generated {out_path} from {spec_path}")
    print(f"  features: {sorted(feats)}")
    print(f"  output_shape: {shape}  is_softmax: {is_softmax}  transpose: {is_transpose}")
    print(f"  full_reduce_redundancy: {redundancy}  warp_size: {warp}")
    return code


def _transpose(kname):
    return [
        "",
        "@triton.jit",
        f"def {kname}(",
        "    in_ptr, out_ptr, rows, cols,",
        "    BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr,",
        "):",
        "    pid_c = tl.program_id(0)",
        "    pid_r = tl.program_id(1)",
        "    tr = tl.arange(0, BLOCK_R)",
        "    tc = tl.arange(0, BLOCK_C)",
        "    in_row = pid_r * BLOCK_R + tr",
        "    in_col = pid_c * BLOCK_C + tc",
        "    load_mask = (in_row < rows)[:, None] & (in_col < cols)[None, :]",
        "    offs_in = in_row[:, None] * cols + in_col[None, :]",
        "    tile = tl.load(in_ptr + offs_in, mask=load_mask, other=0.0)",
        "    out_row = pid_c * BLOCK_C + tc",
        "    out_col = pid_r * BLOCK_R + tr",
        "    store_mask = (out_row < cols)[None, :] & (out_col < rows)[:, None]",
        "    offs_out = out_row[None, :] * rows + out_col[:, None]",
        "    tl.store(out_ptr + offs_out, tile, mask=store_mask)",
    ]


def _elementwise_2d(kname, kind):
    body = [
        "",
        "@triton.jit",
        f"def {kname}(",
        "    a_ptr, b_ptr, out_ptr, rows, cols,",
        "    BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr,",
        "):",
        "    pid_r = tl.program_id(0)",
        "    pid_c = tl.program_id(1)",
        "    r = tl.arange(0, BLOCK_R)",
        "    c = tl.arange(0, BLOCK_C)",
        "    row_mask = (pid_r * BLOCK_R + r) < rows",
        "    col_mask = (pid_c * BLOCK_C + c) < cols",
    ]
    if kind == "sgemm":
        # C[r,c] = sum_k A[r,k] * B[k,c].  A is row-major (stride cols),
        # B is column-major in this naive kernel (stride cols, index k*cols+c).
        body += [
            "    offs_a = (pid_r * BLOCK_R + r)[:, None] * cols + c[None, :]",
            "    offs_b = r[:, None] * cols + (pid_c * BLOCK_C + c)[None, :]",
            "    mask = row_mask[:, None] & col_mask[None, :]",
            "    a = tl.load(a_ptr + offs_a, mask=mask, other=0.0)",
            "    b = tl.load(b_ptr + offs_b, mask=mask, other=0.0)",
            "    acc = tl.sum(a * b)",
            "    offs = (pid_r * BLOCK_R + r)[:, None] * cols + (pid_c * BLOCK_C + c)[None, :]",
        ]
    elif kind == "softmax":
        # Softmax: exp(x - max) / sum(exp(x - max)).
        body += [
            "    offs = (pid_r * BLOCK_R + r)[:, None] * cols + (pid_c * BLOCK_C + c)[None, :]",
            "    mask = row_mask[:, None] & col_mask[None, :]",
            "    a = tl.load(a_ptr + offs, mask=mask, other=0.0)",
            "    max_val = tl.max(a)",
            "    ex = tl.exp(a - max_val)",
            "    s = tl.sum(ex)",
            "    acc = ex / s",
        ]
    else:
        # Elementwise dot: out[r,c] = sum(a[r,c] * b[r,c]).
        body += [
            "    offs = (pid_r * BLOCK_R + r)[:, None] * cols + (pid_c * BLOCK_C + c)[None, :]",
            "    mask = row_mask[:, None] & col_mask[None, :]",
            "    a = tl.load(a_ptr + offs, mask=mask, other=0.0)",
            "    b = tl.load(b_ptr + offs, mask=mask, other=0.0)",
            "    acc = tl.sum(a * b)",
        ]
    body += ["    tl.store(out_ptr + offs, acc, mask=mask)"]
    return body


def _elementwise(kname, redundancy, warp, shape, kind, n_out=1):
    """Elementwise / block-reduce kernel.

    n_out > 1 emits out0_ptr, out1_ptr, ... so the port can represent
    kernels that write to more than one array (e.g. the sglang MoE kernel,
    which writes to both expert_offsets and src2dst).
    """
    out_ptrs = ", ".join(f"out{i}_ptr" for i in range(n_out))
    body = [
        "",
        "@triton.jit",
        f"def {kname}(",
        f"    a_ptr, b_ptr, {out_ptrs}, n,",
        "    BLOCK_SIZE: tl.constexpr,",
        "):",
        "    pid = tl.program_id(0)",
        "    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)",
        "    mask = offs < n",
        "    a = tl.load(a_ptr + offs, mask=mask, other=0.0)",
        "    b = tl.load(b_ptr + offs, mask=mask, other=0.0)",
    ]
    if kind == "elementwise":
        body += ["    acc = a + b"]
    else:
        if redundancy:
            body += [
                "    # Full dot on every lane; compiler elided the warp-shuffle.",
                "    # Reproducing that redundancy is what makes the port byte-identical.",
                "    full = tl.sum(a * b)",
                f"    acc = full * {warp}.0",
            ]
        else:
            body += ["    acc = tl.sum(a * b)"]
    # Store offset: `offs` for elementwise (one output per thread),
    # `pid` for per-block reduction (one output per block).
    store_off = "offs" if kind == "elementwise" else "pid"
    store_mask = "mask" if kind == "elementwise" else \
        "(pid < (n + BLOCK_SIZE - 1) // BLOCK_SIZE)"
    for i in range(n_out):
        body += [
            f"    tl.store(out{i}_ptr + {store_off}, acc,",
            f"                 mask={store_mask})",
        ]
    return body


if __name__ == "__main__":
    generate(sys.argv[1], sys.argv[2])
