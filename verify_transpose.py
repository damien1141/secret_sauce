#!/usr/bin/env python3
"""Verify a 2D tiled-transpose kernel: original CUDA vs generated Triton.

Usage: verify_transpose.py <cubin> <kernel_cu> <rows> <cols>
"""
import json, os, subprocess, sys, ctypes, importlib.util

import torch
import triton

HERE = os.path.dirname(os.path.abspath(__file__))

def main():
    if len(sys.argv) < 5:
        print(f"usage: {sys.argv[0]} <cubin> <kernel_cu> <rows> <cols>")
        return 2
    cubin, kernel_cu = sys.argv[1], sys.argv[2]
    rows, cols = int(sys.argv[3]), int(sys.argv[4])

    # 1. Extract spec
    spec_path = os.path.join(HERE, "spec.json")
    subprocess.run([sys.executable, os.path.join(HERE, "spec_extractor.py"),
                    cubin, spec_path], check=True)

    # 2. Generate
    gen_path = os.path.join(HERE, "generated_kernel.py")
    subprocess.run([sys.executable, os.path.join(HERE, "generate_triton.py"),
                    spec_path, gen_path], check=True)

    # 3. Load generated kernel
    spec_mod = importlib.util.spec_from_file_location("generated_kernel", gen_path)
    mod = importlib.util.module_from_spec(spec_mod)
    spec_mod.loader.exec_module(mod)
    triton_kernel = None
    for name in dir(mod):
        obj = getattr(mod, name)
        if isinstance(obj, triton.runtime.jit.JITFunction):
            triton_kernel = obj
            break
    if triton_kernel is None:
        print("ERROR: no triton kernel found")
        return 1

    # 4. Compile original CUDA kernel
    build = os.path.join(HERE, "build")
    os.makedirs(build, exist_ok=True)
    subprocess.run([
        "/opt/cuda/bin/nvcc", "-shared", "-Xcompiler=-fPIC", "-O3",
        "-arch=sm_80", kernel_cu,
        "-o", os.path.join(build, "libcustom.so"),
    ], check=True)

    lib = ctypes.CDLL(os.path.join(build, "libcustom.so"))
    lib.launch_transpose_kernel.restype = None
    lib.launch_transpose_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.c_int, ctypes.c_int,
    ]

    # 5. Run both on the same input
    TILE = 16
    a = torch.arange(rows * cols, dtype=torch.float32, device="cuda").reshape(rows, cols)
    out_cuda = torch.zeros(cols, rows, dtype=torch.float32, device="cuda")
    out_triton = torch.zeros(cols, rows, dtype=torch.float32, device="cuda")

    lib.launch_transpose_kernel(
        ctypes.cast(a.data_ptr(), ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(out_cuda.data_ptr(), ctypes.POINTER(ctypes.c_float)),
        rows, cols,
    )

    # Generated Triton: program_id(0) = columns, program_id(1) = rows
    grid_x = (cols + TILE - 1) // TILE
    grid_y = (rows + TILE - 1) // TILE
    triton_kernel[(grid_x, grid_y, 1)](
        a.view(-1), out_triton.view(-1), rows, cols,
        BLOCK_R=TILE, BLOCK_C=TILE,
    )

    match = torch.equal(out_cuda, out_triton)
    max_diff = (out_cuda - out_triton).abs().max().item()

    print(f"cubin={os.path.basename(cubin)} kernel={os.path.basename(kernel_cu)}")
    print(f"rows={rows} cols={cols} grid=({grid_x},{grid_y})")
    print(f"CUDA[0,0]   = {out_cuda[0,0].item():.6f}")
    print(f"Triton[0,0] = {out_triton[0,0].item():.6f}")
    print(f"max abs diff = {max_diff:.3e}")
    print(f"byte-for-byte match = {'YES' if match else 'NO'}")
    return 1 if match else 0

if __name__ == "__main__":
    sys.exit(main())