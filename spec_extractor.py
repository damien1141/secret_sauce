#!/usr/bin/env python3
"""Spec extractor: normalizes REA's GPU code-object output + cuobjdump SASS/PTX
into a structured behavioral spec that a Triton generator can consume.

Works on real cubins (SASS-only) and PTX files. Feature detection is SASS-first
because -cubin output contains no PTX, only compiled SASS.

Output shapes detected from the SASS store tail:
  - block_reduce      : one output per block, lane 0 atomicAdds (sum/max)
  - elementwise_2d    : C[y*N+x] = alpha*dot + beta*C  (matmul, softmax)
  - elementwise       : out[idx] = a[idx] op b[idx]        (vec_add)
  - transpose         : shared-memory tiled transpose
  - block_reduce_redundant: full dot on every lane, lane 0 writes
                            (compiler elided the warp-shuffle)

Usage: spec_extractor.py <cubin|ptx> <spec_out.json>
"""
import json, subprocess, sys, re

def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout

def main():
    src, out = sys.argv[1], sys.argv[2]
    is_ptx = src.endswith(".ptx")

    if is_ptx:
        with open(src) as f:
            ptx = f.read()
        sass = ""
    else:
        sass = run(["/opt/cuda/bin/cuobjdump", "-sass", src])
        ptx  = run(["/opt/cuda/bin/cuobjdump", "-ptx",  src])

    # Kernel name: SASS first, PTX fallback
    kname = "unknown"
    m = re.search(r"Function : (\S+)", sass)
    if m:
        kname = m.group(1)
    else:
        m = re.search(r"\.entry\s+(\w+)", ptx)
        if m:
            kname = m.group(1)

    # SM target from PTX, or from SASS headerflags
    target = "sm_unknown"
    m = re.search(r"\.target\s+(\S+)", ptx)
    if m:
        target = m.group(1)
    elif "EF_CUDA_SM80" in sass:
        target = "sm_80"

    # Shared memory bytes — PTX .shared[N] first, SASS BSSY immediate fallback
    shared_bytes = 0
    for line in ptx.splitlines():
        if ".shared" in line and "[" in line:
            n = re.search(r"\[(\d+)\]", line)
            if n:
                elem = 4 if ".f32" in line or ".b32" in line else 1
                shared_bytes = int(n.group(1)) * elem
    if not shared_bytes:
        m = re.search(r"BSSY\s+\w+,\s*0x([0-9a-fA-F]+)", sass)
        if m:
            shared_bytes = int(m.group(1), 16)

    # Feature detection — SASS-first (works on cubins with no PTX)
    feats = []
    if "STS" in sass or ".shared" in ptx:            feats.append("shared_memory")
    if "BSYNC" in sass or "BAR.SYNC" in sass or "bar.sync" in ptx: feats.append("bar_sync")
    if "SHFL" in sass or "shfl.sync.down" in ptx:    feats.append("shfl_down_sync")
    if "FFMA" in sass or "fma." in ptx:              feats.append("fma")
    if "LDG" in sass or "ld.global" in ptx:          feats.append("ldg")
    if "STS" in sass or "st.shared" in ptx:          feats.append("sts")
    if "ATOM" in sass or "atom." in ptx:             feats.append("atomic_reduction")
    if "ld.param" in ptx:                            feats.append("params")
    if "st.global" in ptx or "STG" in sass:          feats.append("st_global")
    if ".pred" in ptx:                               feats.append("predicates")
    if "IMAD" in sass or "mad.lo" in ptx:            feats.append("imad")
    if "ULDC" in sass:                               feats.append("uniform_load")
    if "ISETP" in sass:                              feats.append("isetp")
    if "LDS" in sass or "ld.shared" in ptx:          feats.append("lds")

    # --- cudad memory-space features ---
    # cudad recovers per-access memory space/kind from the SASS, which the
    # opcode bag cannot: it distinguishes a shared-memory kernel (transpose,
    # reduce, custom) from a pure-global one (sgemm, sgemv, softmax), and
    # flags atomic reductions.  This is the signal that lets us classify
    # data-layout semantics the opcode set alone cannot.
    CUDAD_MEM = None
    try:
        import cudad
        d = cudad.analyze(src)
        if d is not None:
            mem = d["functions"][0]["memory"]
            CUDAD_MEM = {
                "by_space": mem.get("by_space", {}),
                "by_kind": mem.get("by_kind", {}),
                "access_count": mem.get("access_count", 0),
                "dynamic_address_accesses": mem.get("dynamic_address_accesses", 0),
            }
            for space, n in mem.get("by_space", {}).items():
                feats.append(f"mem_{space}")
            for kind, n in mem.get("by_kind", {}).items():
                feats.append(f"op_{kind}")
            # Number of distinct output arrays: count the distinct write roots
            # in cudad's memory inventory.  Single-output kernels have 1.
            roots = {int(a["root"].split("_")[-1])
                     for a in mem.get("accesses", [])
                     if a["kind"] == "store"
                     and a.get("root", "").startswith("param_word_")}
            n_out = max(1, len(roots))
    except Exception:
        pass

    # --- Behavioral signature detection ---
    # Output shape, detected from the SASS store tail:
    #   atomic_reduction  -> block_reduce  (one output per block, lane 0 atomicAdds)
    #   STG + 2D index     -> elementwise_2d (C[y*N+x] = alpha*dot + beta*C)
    #   STG + 1D index     -> elementwise   (out[idx] = a[idx] op b[idx])
    #   sts/lds/bar_sync   -> transpose     (shared-memory tiled)
    full_reduce_redundancy = False
    warp_size = 32
    if "FFMA" in sass and "bar_sync" in feats and "atomic_reduction" not in feats \
       and "shfl_down_sync" not in feats:
        # Full-reduction redundancy: every lane computes the FULL dot product
        # (FFMA chains over all elements), then lane 0 writes. The compiler
        # eliminates the warp-shuffle because all lanes hold identical values,
        # so no SHFL instruction appears — but the behavior is still
        # result = 32 * dot. Detect this from the SASS structure.
        ffma_count = len(re.findall(r"FFMA\s+", sass))
        if ffma_count >= 32:
            full_reduce_redundancy = True

    # Full-reduction redundancy (256 FFMA = 32 lanes x 8 elements, lane 0
    # writes) takes precedence over the transpose heuristic: the redundant
    # kernel has sts/lds/bar_sync too, but it is NOT a transpose.
    if full_reduce_redundancy:
        output_shape = "block_reduce"
    elif "atomic_reduction" in feats or "op_reduction" in feats:
        # cudad flags a global atomicAdd as op_reduction — this is the
        # authoritative signal for a per-block reduction, and it overrides
        # the shared-memory heuristic (reduce_v4 has sts/lds/bar_sync too).
        output_shape = "block_reduce"
    elif "sts" in feats and "lds" in feats and "bar_sync" in feats:
        output_shape = "transpose"
    elif "st_global" in feats:
        # Distinguish 2D (matmul/softmax) from 1D (vec_add) stores by whether
        # the SASS computes a per-thread accumulator (FFMA chain) vs a single
        # per-element op. A 2D store has FMUL/FFMA at the store tail.
        output_shape = "elementwise_2d" if ("FFMA" in sass or "FMUL" in sass) else "elementwise"
    else:
        output_shape = "block_reduce"

    # Register count from .nv.info via rea
    reg_info = {}
    if not is_ptx:
        try:
            raw = run(["/home/solis/.bun/bin/rea-agents", "inspect-gpu-code-objects",
                       src, "--analysis", "kernels", "--format", "json"])
            data = json.loads(raw)
            for k in data.get("normalized_result", {}).get("kernels", []):
                reg_info = k.get("registers", {})
                break
        except Exception:
            pass

    spec = {
        "kernel": kname,
        "target": target,
        "shared_bytes": shared_bytes,
        "features": feats,
        "registers": reg_info,
        "output_shape": output_shape,
        "full_reduce_redundancy": full_reduce_redundancy,
        "warp_size": warp_size,
        "sass": sass,
        "ptx": ptx,
        "cudad_memory": CUDAD_MEM,
        "n_out": n_out,
    }

    with open(out, "w") as f:
        json.dump(spec, f, indent=2)
    print(f"spec written to {out}")
    print(f"  kernel={kname} target={target} shared={shared_bytes}B")
    print(f"  features={feats}")
    print(f"  output_shape={output_shape} full_reduce_redundancy={full_reduce_redundancy}")

if __name__ == "__main__":
    main()