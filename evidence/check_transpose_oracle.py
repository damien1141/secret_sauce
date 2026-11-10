"""Check: does the generated Triton transpose match torch.transpose (independent oracle)?"""
import json, os, subprocess, sys, importlib.util, ctypes
import torch, triton
HERE=os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.'
# regenerate from the transpose cubin
subprocess.run([sys.executable, "spec_extractor.py", "cublas_fatbin.2.sm_80.cubin", "spec.json"], check=True, capture_output=True)
subprocess.run([sys.executable, "generate_triton.py", "spec.json", "generated_kernel.py"], check=True, capture_output=True)
spec_mod=importlib.util.spec_from_file_location("g","generated_kernel.py")
mod=importlib.util.module_from_spec(spec_mod); spec_mod.loader.exec_module(mod)
k=[v for v in vars(mod).values() if isinstance(v,triton.runtime.jit.JITFunction)][0]
TILE=16
for (rows,cols) in [(16,128),(128,16),(64,64),(16,16),(128,128),(33,97)]:
    a=torch.arange(rows*cols,dtype=torch.float32,device="cuda").reshape(rows,cols)
    out=torch.zeros(cols,rows,dtype=torch.float32,device="cuda")
    gx=(cols+TILE-1)//TILE; gy=(rows+TILE-1)//TILE
    k[(gx,gy,1)](a.view(-1),out.view(-1),rows,cols,BLOCK_R=TILE,BLOCK_C=TILE)
    ref=a.transpose(0,1)
    ok=torch.equal(out,ref)
    print(f"{rows}x{cols}: triton==torch.transpose: {ok}  maxdiff={(out-ref).abs().max().item():.3e}")
