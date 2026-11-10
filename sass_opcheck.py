#!/usr/bin/env python3
"""sass_opcheck.py — differential op test: GPU vs emulator, per SASS idiom.

Compiles a micro-kernel at sm_80 whose bodies are the exact integer idioms
ptxas emits for real code (signed division, modulo, LEA address math), runs it
on the GPU and in sass_emu.py over randomized inputs, and reports every case
where the two disagree.

This exists because guessing SASS semantics from the disassembly is how wrong
answers get shipped: LEA.HI / LEA.HI.SX32 / IMAD.HI.U32 / SHF carry subtleties
that only a differential run settles.

usage: ./sass_opcheck.py [--trials N] [--seed S]
"""
import ctypes
import os
import random
import struct
import subprocess
import sys

import sass_emu

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(HERE, "build")
CU = "/opt/cuda/bin/nvcc"

SOURCE = r"""
// Idioms the emulator claims to support.  Each line is a distinct SASS shape:
// shifts (SHF), 32/64-bit multiply (IMAD/IMAD.WIDE), signed-high multiply
// (IMAD.HI), select (SEL/IMNMX), bit logic (LOP3), negate/abs (IABS), and
// 64-bit address arithmetic (LEA/IMAD.WIDE) with 2-D indexing.
extern "C" __global__ void ishift(int a, int b, int* out) {
    out[0]  = a >> 1;
    out[1]  = a >> 3;
    out[2]  = (unsigned)a >> 3;
    out[3]  = a << 3;
    out[4]  = (a << 3) + b;
    out[5]  = ((unsigned)a >> 5) | (b << 2);
    out[6]  = a + (b - a) / 2;    // the lower_bound mid idiom
    out[7]  = (b - a) / 2;
    out[8]  = (b + a) / 2;
    out[9]  = a ^ b;
    out[10] = a / 2 + b / 2;      // two halvings, no /4 fixup
    out[11] = (a * 3) + (b * 5);
}

extern "C" __global__ void imisc(int a, int b, int* out) {
    out[0]  = a * b;
    out[1]  = a * 1000 + b;
    out[2]  = (int)(((long long)a * b) >> 32);
    out[3]  = (unsigned)(((unsigned long long)(unsigned)a * (unsigned)b) >> 32);
    out[4]  = a < b ? a : b;
    out[5]  = (a & 31) == 0;
    out[6]  = a ^ b;
    out[7]  = (a & 0xf0f0f0f0) | (b & 0x0f0f0f0f);
    out[8]  = ~a;
    out[9]  = (a & 0xff) * (b & 0xff);
    out[10] = abs(a - b);
    out[11] = a == b ? 12345 : -1;
}

extern "C" __global__ void iaddr(int idx, int n, int* base, int* out) {
    // 64-bit address arithmetic across the LEA / IMAD.WIDE family, including
    // reversed indexing and multi-dimensional stride math.
    out[0] = base[idx];
    out[1] = base[idx + 7];
    out[2] = base[n - idx - 1];
    out[3] = base[idx] + base[n - idx - 1];
    out[4] = base[idx * 2 + 1];
    out[5] = base[(idx + n) / 2];
}
"""


def build():
    os.makedirs(BUILD, exist_ok=True)
    src = os.path.join(BUILD, "opcheck.cu")
    with open(src, "w") as f:
        f.write(SOURCE)
    cubin = os.path.join(BUILD, "opcheck.cubin")
    subprocess.run([CU, "-arch=sm_80", "-cubin", "-O3", "-o", cubin, src],
                   check=True, capture_output=True)
    return cubin


class Gpu:
    def __init__(self):
        from cuda.bindings import driver as cp
        self.cp = cp
        cp.cuInit(0)
        err, dev = cp.cuDeviceGet(0)
        err, ctx = cp.cuDevicePrimaryCtxRetain(dev)
        cp.cuCtxSetCurrent(ctx)
        lib = ctypes.CDLL("libcuda.so.1")
        lib.cuLaunchKernel.restype = ctypes.c_int
        lib.cuLaunchKernel.argtypes = [
            ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        self.lib = lib
        self._keep = []

    def run(self, cubin, func, args_i32, nout, nbase=None, base_vals=None):
        cp = self.cp
        err, mod = cp.cuModuleLoad(cubin.encode())
        assert int(err) == 0, err
        err, fn = cp.cuModuleGetFunction(mod, func.encode())
        assert int(err) == 0, err
        err, dout = cp.cuMemAlloc(4 * nout)
        assert int(err) == 0, err
        cp.cuMemsetD8(dout, 0, 4 * nout)
        err, dbase = cp.cuMemAlloc(4 * max(1, nbase or 1))
        assert int(err) == 0, err
        if base_vals:
            cp.cuMemcpyHtoD(dbase, struct.pack("<%di" % len(base_vals), *base_vals),
                            4 * len(base_vals))
        params = []
        for kind, v in args_i32:
            if kind == "ptr":
                params.append(("ptr", int(dout if v == "out" else dbase)))
            else:
                params.append(("i32", v))
        buf = bytearray()
        offs = []
        for kind, v in params:
            natural = 8 if kind == "ptr" else 4
            while len(buf) % natural:
                buf += b"\x00"
            offs.append(len(buf))
            buf += (int(v) & ((1 << (8 * natural)) - 1)).to_bytes(natural, "little")
        cbuf = (ctypes.c_ubyte * len(buf)).from_buffer(buf)
        cbase = ctypes.addressof(cbuf)
        kp = (ctypes.c_void_p * len(offs))(*[cbase + o for o in offs])
        self._keep = [cbuf, kp]
        r = self.lib.cuLaunchKernel(ctypes.c_void_p(int(fn)), 1, 1, 1, 1, 1, 1,
                                    0, None, kp, None)
        assert r == 0, f"launch {r}"
        err, = cp.cuCtxSynchronize()
        assert int(err) == 0, err
        out = (ctypes.c_ubyte * (4 * nout))()
        err, = cp.cuMemcpyDtoH(out, dout, 4 * nout)
        assert int(err) == 0, err
        return bytes(out)


def emu(cubin, func, args_i32, nout, nbase=None, base_vals=None):
    m = sass_emu.SASSMachine(sass_emu.sass_of(cubin), func=func)
    m.set_launch((1, 1, 1), (1, 1, 1))
    OUT, BASE = 0x20000000, 0x30000000
    m.load_bytes(OUT, b"\x00" * (4 * nout))
    nb = 4 * max(1, nbase or 1)
    m.load_bytes(BASE, struct.pack("<%di" % (nbase or 1), *base_vals)[:nb]
                 if base_vals else b"\x00" * nb)
    params = []
    for kind, v in args_i32:
        if kind == "ptr":
            params.append(("ptr", OUT if v == "out" else BASE))
        else:
            params.append(("i32", v & 0xFFFFFFFF))
    m.set_params(params)
    m.run()
    return m.read_bytes(OUT, 4 * nout)


def main(argv):
    trials = 40
    seed = 1
    for i, a in enumerate(argv):
        if a == "--trials":
            trials = int(argv[i + 1])
        if a == "--seed":
            seed = int(argv[i + 1])
    cubin = build()
    gpu = Gpu()
    rng = random.Random(seed)
    bad = 0
    total = 0

    skipped = []

    def check(func, args, nout, label, nbase=None, base_vals=None):
        nonlocal bad, total
        g = gpu.run(cubin, func, args, nout, nbase, base_vals)
        try:
            e = emu(cubin, func, args, nout, nbase, base_vals)
        except sass_emu.Unsupported as exc:
            # An honest refusal is not a wrong answer: count it as skipped and
            # report it, so "unsupported" never hides behind a pass rate.
            skipped.append(f"{label}: {exc}")
            return
        total += 1
        if g != e:
            bad += 1
            gi = struct.unpack("<%di" % nout, g)
            ei = struct.unpack("<%di" % nout, e)
            diffs = [f"[{k}] gpu={gi[k]} emu={ei[k]}" for k in range(nout)
                     if gi[k] != ei[k]]
            print(f"  MISMATCH {label}: {', '.join(diffs[:4])}")

    for t in range(trials):
        a = rng.choice([rng.randint(-2**31, 2**31 - 1), rng.randint(-50, 50),
                        rng.randint(0, 1000), -1, 0, 7, -7])
        b = rng.choice([x for x in range(-40, 41) if x != 0] +
                       [rng.randint(-2**31, 2**31 - 1) or 3])
        check("ishift", [("i32", a), ("i32", b), ("ptr", "out")], 12,
              f"ishift a={a} b={b}")
        check("imisc", [("i32", a), ("i32", b), ("ptr", "out")], 12,
              f"imisc a={a} b={b}")
        n = 64
        idx = rng.randint(0, 16)
        base_vals = [rng.randint(-2**31, 2**31 - 1) for _ in range(n)]
        check("iaddr", [("i32", idx), ("i32", n), ("ptr", "base"), ("ptr", "out")],
              6, f"iaddr idx={idx}", nbase=n, base_vals=base_vals)

    print(f"{total - bad}/{total} idiom cases byte-identical"
          f"  ({bad} mismatches)")
    if skipped:
        seen = set()
        print(f"{len(skipped)} case(s) skipped as unsupported:")
        for sk in skipped:
            key = sk.split(": ", 1)[1]
            if key in seen:
                continue
            seen.add(key)
            print(f"  - {key}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
