import os
import sys
import ctypes
import subprocess
import torch

TILE = 256

os.makedirs("/tmp/build", exist_ok=True)
subprocess.run(
    ["/opt/cuda/bin/nvcc", "-shared", "-Xcompiler=-fPIC", "-O3",
     "-arch=sm_80", "/tmp/kernel.cu", "-o", "/tmp/build/libcustom.so"],
    check=True,
)

lib = ctypes.CDLL("/tmp/build/libcustom.so")
lib.launch_custom_kernel.restype = None
lib.launch_custom_kernel.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
]

import triton
import triton.language as tl

@triton.jit
def triton_kernel(
    a_ptr, b_ptr, out_ptr, n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n

    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)

    # FAITHFUL REPLICATION of the CUDA kernel's algorithm structure.
    #
    # The CUDA kernel has every thread loop over ALL 256 tile elements,
    # so every lane ends up holding the SAME full dot product. The
    # shfl_down_sync chain (offsets 16/8/4/2/1) then sums 32 identical
    # copies across the warp — which is exactly 32 * full.
    #
    # Replicating both steps: full dot, then the warp-wide sum of 32 copies.
    full = tl.sum(a * b)
    acc = full * 32.0

    if pid * BLOCK_SIZE < n:
        tl.store(out_ptr + pid, acc, mask=(pid < (n + BLOCK_SIZE - 1) // BLOCK_SIZE))


def run(n: int, blocks: int, tile: int):
    a = torch.empty(n, dtype=torch.float32, device="cuda")
    b = torch.empty(n, dtype=torch.float32, device="cuda")
    a.copy_(torch.arange(n, dtype=torch.float32, device="cuda") % 100 * 0.5)
    b.copy_((torch.arange(n, dtype=torch.float32, device="cuda") * 3) % 100 * 0.25)

    out_cuda = torch.zeros(blocks, dtype=torch.float32, device="cuda")
    out_triton = torch.zeros(blocks, dtype=torch.float32, device="cuda")

    lib.launch_custom_kernel(
        ctypes.cast(a.data_ptr(), ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(b.data_ptr(), ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(out_cuda.data_ptr(), ctypes.POINTER(ctypes.c_float)),
        n,
    )

    triton_kernel[(blocks, 1, 1)](a, b, out_triton, n, BLOCK_SIZE=tile)

    match = torch.equal(out_cuda, out_triton)
    max_diff = (out_cuda - out_triton).abs().max().item()

    print(f"n={n} blocks={blocks}")
    print(f"CUDA   out[0] = {out_cuda[0].item():.6f}")
    print(f"Triton out[0] = {out_triton[0].item():.6f}")
    print(f"max abs diff = {max_diff:.3e}")
    print(f"byte-for-byte match = {'YES' if match else 'NO'}")
    return 1 if match else 0


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
    blocks = (n + TILE - 1) // TILE
    sys.exit(run(n, blocks, TILE))