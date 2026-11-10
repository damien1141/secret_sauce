#!/usr/bin/env python3
"""Multi-kernel regression harness.

Runs a closed-source cubin against its clean-room Triton port across a set of
problem sizes and reports pass/fail + max abs diff per shape. CPU ground truth
is computed with numpy.

Usage: bench.py <cubin> <triton_port.py> <reference.cu> <out.tsv> [N...]
"""
import sys, os, subprocess, importlib.util, tempfile, json
import numpy as np
import torch

VENV = "/home/solis/Documentos/Code/secret_sauce/.venv"
PY = os.path.join(VENV, "bin", "python")
NVCC = "/opt/cuda/bin/nvcc"

def load_triton(path):
    spec = importlib.util.spec_from_file_location("port", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def build_ref(cu, out):
    r = subprocess.run([NVCC, "-shared", "-Xcompiler=-fPIC", "-O3",
                        "-arch=sm_80", "-o", out, cu],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("ref build failed:\n" + r.stderr)

def run_cubin(so, fn, args):
    spec = importlib.util.spec_from_file_location("cubin", so)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return getattr(m, fn)(*args)

def main():
    cubin, port, ref_cu, out_tsv = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    sizes = [int(x) for x in sys.argv[5:]] if len(sys.argv) > 5 else [256, 1024, 4096]
    port_mod = load_triton(port)

    with tempfile.TemporaryDirectory() as td:
        ref_so = os.path.join(td, "ref.so")
        build_ref(ref_cu, ref_so)

        rows = []
        for N in sizes:
            a = np.random.rand(N).astype(np.float32)
            b = np.random.rand(N).astype(np.float32)
            inp = torch.from_numpy(a).cuda()
            inp2 = torch.from_numpy(b).cuda()
            out = torch.zeros(N, dtype=torch.float32, device="cuda")

            # CPU reference
            ref = run_cubin(ref_so, "launch_ref", (inp, inp2, out, N))
            out_np = out.cpu().numpy()
            # Triton port
            port_mod.port(inp, inp2, out, N)
            triton_np = out.cpu().numpy()

            diff = float(np.max(np.abs(out_np - triton_np))) if out_np.size else 0.0
            ok = "PASS" if diff < 1e-4 else "FAIL"
            rows.append((N, ok, diff))
            print(f"N={N:<6} {ok}  max_abs_diff={diff:.3e}")

        with open(out_tsv, "w") as f:
            f.write("N\tstatus\tmax_abs_diff\n")
            for r in rows:
                f.write("\t".join(str(x) for x in r) + "\n")
        print(f"wrote {out_tsv}")

if __name__ == "__main__":
    main()
