import os, subprocess, sys, ctypes
import torch
# compile the reference transpose_kernel.cu and check vs torch.transpose
os.makedirs("/tmp/b2", exist_ok=True)
subprocess.run(["/opt/cuda/bin/nvcc","-shared","-Xcompiler=-fPIC","-O3","-arch=sm_80",
                "transpose_kernel.cu","-o","/tmp/b2/lib.so"], check=True)
lib=ctypes.CDLL("/tmp/b2/lib.so")
lib.launch_transpose_kernel.restype=None
lib.launch_transpose_kernel.argtypes=[ctypes.POINTER(ctypes.c_float),ctypes.POINTER(ctypes.c_float),ctypes.c_int,ctypes.c_int]
TILE=16
for (rows,cols) in [(16,128),(128,16),(64,64),(33,97)]:
    a=torch.arange(rows*cols,dtype=torch.float32,device="cuda").reshape(rows,cols)
    out=torch.zeros(cols,rows,dtype=torch.float32,device="cuda")
    lib.launch_transpose_kernel(ctypes.cast(a.data_ptr(),ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(out.data_ptr(),ctypes.POINTER(ctypes.c_float)),rows,cols)
    ref=a.transpose(0,1)
    print(f"{rows}x{cols}: ref==torch.transpose: {torch.equal(out,ref)} maxdiff={(out-ref).abs().max().item():.3e}")
