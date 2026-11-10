#!/usr/bin/env python3
"""SASS signature corpus: extract a normalized opcode fingerprint from every
CUDA kernel in /tmp/CUDA_Kernel_Samples and match closed-source cubins.

The fingerprint is a bag-of-opcodes + reduction-style + store-shape tuple.
It is computed from SASS alone (cuobjdump -sass), so it works on closed-source
cubins with no PTX and no source.  Matching is heuristic: a high score means
the closed-source kernel shares the same reduction structure as a known
open-source sample - it does NOT prove equivalence.

Usage: corpus.py <cubin>            -> print best-matching sample
       corpus.py --build <dir>      -> build the corpus index from .cu files
"""
import os, re, subprocess, sys, json, collections

SAMPLES = "/tmp/CUDA_Kernel_Samples"
CUOBJDUMP = "/opt/cuda/bin/cuobjdump"

# Canonical opcode list (SASS mnemonics, op-size stripped).
OP_RE = re.compile(r"\b([A-Z][A-Z0-9_]{1,8})\b")


def sass_of(path):
    if path.endswith(".ptx"):
        with open(path) as f:
            return "", f.read()
    sass = subprocess.run([CUOBJDUMP, "-sass", path],
                          capture_output=True, text=True).stdout
    ptx = subprocess.run([CUOBJDUMP, "-ptx", path],
                         capture_output=True, text=True).stdout
    return sass, ptx


def fingerprint(sass, ptx):
    ops = collections.Counter(m for m in OP_RE.findall(sass) if m not in
                              ("RZ", "PT", "URZ", "UPT", "WIDE", "AND", "GE",
                               "NE", "GT", "LT", "OR", "LOP3", "PLOP3"))
    feats = set()
    if "STS" in sass or ".shared" in ptx:            feats.add("shared")
    if "BSYNC" in sass or "bar.sync" in ptx:         feats.add("bar_sync")
    if "SHFL" in sass or "shfl.sync.down" in ptx:    feats.add("shfl_down")
    if "SHFLX" in sass or "shfl.sync.xor" in ptx:    feats.add("shfl_xor")
    if "ATOM" in sass or "atom." in ptx:             feats.add("atomic")
    if "FFMA" in sass or "fma." in ptx:              feats.add("ffma")
    if "FMUL" in sass or "fmul." in ptx:             feats.add("fmul")
    if "FADD" in sass or "fadd." in ptx:             feats.add("fadd")
    if "FMNM" in sass or "fmin." in ptx or "fmax." in ptx: feats.add("fmaxmin")
    if "EXP" in sass or "exp." in ptx:               feats.add("exp")
    if "LDG" in sass or "ld.global" in ptx:          feats.add("ldg")
    if "STG" in sass or "st.global" in ptx:          feats.add("stg")
    if "IMAD" in sass or "mad.lo" in ptx:            feats.add("imad")
    if "ISETP" in sass or "setp." in ptx:            feats.add("isetp")
    if "ULDC" in sass:                               feats.add("uniform_load")
    if "STS" in sass or "st.shared" in ptx:          feats.add("sts")
    if "LDS" in sass or "ld.shared" in ptx:          feats.add("lds")
    # Reduction style.
    if "atomic" in feats:
        style = "block_reduce_atomic"
    elif "shfl_down" in feats or "shfl_xor" in feats:
        style = "warp_shuffle_reduce"
    elif "ffma" in feats and "bar_sync" in feats:
        style = "shared_tile"
    elif "ffma" in feats:
        style = "dot_product"
    elif "exp" in feats or "fmaxmin" in feats:
        style = "nonlinear_elementwise"
    else:
        style = "elementwise"
    return {"ops": dict(ops), "feats": sorted(feats), "style": style}


def find_cu_files(root):
    out = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.endswith(".cu"):
                out.append(os.path.join(dp, fn))
    return out


def build(dirpath):
    index = {}
    for cu in find_cu_files(dirpath):
        # Compile each .cu to a cubin so we can fingerprint the SASS.
        out = "/tmp/_corpus_" + os.path.basename(cu) + ".cubin"
        r = subprocess.run([CUOBJDUMP.replace("cuobjdump", "nvcc"),
                            "-cubin", "-arch=sm_80", "-o", out, cu],
                           capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(out):
            continue
        sass, ptx = sass_of(out)
        fp = fingerprint(sass, ptx)
        # Pull the kernel name from the symbol table.
        kname = "unknown"
        m = re.search(r"Function : (\S+)", sass)
        if m:
            kname = m.group(1)
        index[os.path.relpath(cu, dirpath)] = {
            "kernel": kname, "fp": fp, "cubin": out,
        }
    return index


def score(a, b):
    fa, fb = set(a["feats"]), set(b["feats"])
    shared = fa & fb
    only_a = fa - fb
    only_b = fb - fa
    # Style match is the dominant signal; shared features are secondary.
    s = 0.0
    if a["style"] == b["style"]:
        s += 10.0
    s += len(shared) * 1.0
    s -= len(only_a) * 0.5 + len(only_b) * 0.5
    return s


def load(path="/tmp/corpus_index.json"):
    with open(path) as f:
        return json.load(f)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    if sys.argv[1] == "--build":
        d = sys.argv[2] if len(sys.argv) > 2 else SAMPLES
        idx = build(d)
        with open("/tmp/corpus_index.json", "w") as f:
            json.dump(idx, f, indent=2)
        print(f"built corpus: {len(idx)} kernels -> /tmp/corpus_index.json")
        for k, v in idx.items():
            print(f"  {k:40s} {v['kernel']:45s} {v['fp']['style']}")
        return 0
    target = sys.argv[1]
    with open("/tmp/corpus_index.json") as f:
        idx = json.load(f)
    sass, ptx = sass_of(target)
    tfp = fingerprint(sass, ptx)
    print(f"target style={tfp['style']} feats={tfp['feats']}")
    ranked = sorted(idx.items(), key=lambda kv: -score(tfp, kv[1]["fp"]))
    for path, info in ranked[:5]:
        sc = score(tfp, info["fp"])
        print(f"  {sc:6.2f}  {path:40s} {info['kernel']:40s} {info['fp']['style']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
