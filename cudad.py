#!/usr/bin/env python3
"""Wrapper around the `cudad` SASS decompiler (vanehl-arial/cudad).

cudad reads `cuobjdump -sass` text and recovers:
  - ABI parameter layout with types (ptr64/word32, pointee_type, c_type)
  - Memory-space access counts (global/shared/const/local)
  - CFG metrics, opcode coverage, unresolved-control hazards

This is the structured-facts layer between raw SASS and the Triton
generator.  It is independent evidence: cudad's ABI recovery and our
KPARAM parser agree on every closed-source target we have tested.

Usage: cudad.py <cubin> <out.json>
"""
import json, os, subprocess, tempfile

CIDAD = "/tmp/cudad/target/release/main"
CUOBJDUMP = "/opt/cuda/bin/cuobjdump"


def analyze(cubin, out_json=None):
    """Run cudad on a cubin; return the analysis dict (or None on failure)."""
    sass = subprocess.run([CUOBJDUMP, "-sass", cubin],
                          capture_output=True, text=True).stdout
    if "code for" not in sass:
        return None
    with tempfile.TemporaryDirectory() as td:
        sp = os.path.join(td, "k.sass")
        open(sp, "w").write(sass)
        jf = os.path.join(td, "out.json")
        r = subprocess.run([CIDAD, "--input", sp, "--analysis-json",
                            "--output", jf],
                           capture_output=True, text=True)
        if r.returncode != 0:
            return None
        data = json.load(open(jf))
    if out_json:
        json.dump(data, open(out_json, "w"), indent=2)
    return data


def abi_params(data):
    """Return [(param_idx, kind, c_type, pointee_type)] from cudad ABI."""
    f = data["functions"][0]
    return [(p["param_idx"], p["kind"], p.get("c_type"), p.get("pointee_type"))
            for p in f["abi"]["parameter_aliases"]]


def memory_summary(data):
    f = data["functions"][0]
    return {k: v for k, v in f["memory"].items()
            if not isinstance(v, (list, dict))}


if __name__ == "__main__":
    import sys
    d = analyze(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    if d is None:
        print("cudad analysis failed"); sys.exit(1)
    print(f"function={d['functions'][0]['name']} "
          f"instr={d['functions'][0]['instruction_count']}")
    for idx, kind, ctype, ptype in abi_params(d):
        print(f"  param[{idx:2d}] {kind:8s} c_type={ctype} pointee={ptype}")
    print("memory:", memory_summary(d))
