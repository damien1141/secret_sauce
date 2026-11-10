#!/usr/bin/env python3
"""sass_verify.py — prove the SASS emulator against real hardware.

For each case:
  1. load the cubin on the GPU (CUDA driver API) and launch it,
  2. run the SAME cubin's SASS through sass_emu.py,
  3. compare every output byte.

The emulator gets the launch arguments only through the kernel-parameter
constant bank, and the comparison is on raw bytes, not on a tolerance.
Every case also carries an independent numpy/torch reference so a case that
"passes" because both sides produced garbage is visible.

usage:
  ./sass_verify.py                 # all cases
  ./sass_verify.py vec_add moe     # named cases
"""
import ctypes
import os
import struct
import subprocess
import sys

import numpy as np

import sass_emu

HERE = os.path.dirname(os.path.abspath(__file__))
CUOBJDUMP = "/opt/cuda/bin/cuobjdump"

DT = {"f32": (4, "f"), "i32": (4, "i"), "i64": (8, "q")}


def encode(ty, values):
    if ty == "f32":
        return struct.pack("<%df" % len(values), *values)
    if ty == "i32":
        return struct.pack("<%di" % len(values), *values)
    if ty == "i64":
        return struct.pack("<%dq" % len(values), *values)
    if ty == "raw":
        return bytes(values)
    raise ValueError(ty)


def decode(ty, data, n):
    if ty == "f32":
        return np.frombuffer(data[:4 * n], dtype="<f4")
    if ty == "i32":
        return np.frombuffer(data[:4 * n], dtype="<i4")
    if ty == "i64":
        return np.frombuffer(data[:8 * n], dtype="<i8")
    if ty == "raw":
        return np.frombuffer(data[:n], dtype="u1")
    raise ValueError(ty)


# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------
class Gpu:
    """Context/module/memory via cuda-python; cuLaunchKernel via libcuda.

    cuda-python 13.x's cuLaunchKernel rejects every documented hStream form
    ("an integer is required"), so the launch is issued through
    libcuda.so.1 directly with an explicit signature.  Both share the same
    primary context, so the module and buffers are interchangeable.
    """

    def __init__(self):
        from cuda.bindings import driver as cp
        import ctypes.util

        self.cp = cp
        cp.cuInit(0)
        err, ctx = cp.cuCtxGetCurrent()
        if int(err) != 0 or int(ctx) == 0:
            err, dev = cp.cuDeviceGet(0)
            assert int(err) == 0, err
            err, ctx = cp.cuDevicePrimaryCtxRetain(dev)
            assert int(err) == 0, err
            err, = cp.cuCtxSetCurrent(ctx)
            assert int(err) == 0, err
        self.ctx = ctx

        lib = ctypes.CDLL("libcuda.so.1")
        lib.cuLaunchKernel.restype = ctypes.c_int
        lib.cuLaunchKernel.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ]
        self.cu = lib
        self._keep = []

    def module(self, path):
        err, mod = self.cp.cuModuleLoad(path.encode())
        assert int(err) == 0, f"cuModuleLoad {path}: {err}"
        return mod

    def function(self, mod, name):
        err, f = self.cp.cuModuleGetFunction(mod, name.encode())
        assert int(err) == 0, f"cuModuleGetFunction {name}: {err}"
        return f

    def memcpy_htod(self, data):
        cp = self.cp
        err, d = cp.cuMemAlloc(max(len(data), 1))
        assert int(err) == 0, err
        err, = cp.cuMemcpyHtoD(d, data, len(data))
        assert int(err) == 0, err
        return d

    def memcpy_dtoh(self, dptr, nbytes):
        buf = (ctypes.c_ubyte * max(nbytes, 1))()
        err, = self.cp.cuMemcpyDtoH(buf, dptr, nbytes)
        assert int(err) == 0, err
        return bytes(buf)[:nbytes]

    def launch(self, f, grid, block, params, smem=0):
        """params: [(kind, value)] packed exactly the way the driver expects."""
        buf = bytearray()
        offs = []
        for kind, v in params:
            natural = 8 if kind in ("ptr", "i64") else 4
            while len(buf) % natural:
                buf += b"\x00"
            offs.append(len(buf))
            if kind in ("ptr", "i64"):
                buf += int(v).to_bytes(8, "little", signed=False)
            elif kind == "f32":
                buf += struct.pack("<f", v)
            elif kind == "i8":
                buf += bytes([int(v) & 0xFF]) + b"\x00\x00\x00"
            else:
                buf += int(v).to_bytes(4, "little", signed=(int(v) < 0))
        cbuf = (ctypes.c_ubyte * max(1, len(buf))).from_buffer(buf)
        base = ctypes.addressof(cbuf)
        kp = (ctypes.c_void_p * max(1, len(offs)))()
        for i, o in enumerate(offs):
            kp[i] = base + o
        self._keep = [cbuf, kp, buf]
        r = self.cu.cuLaunchKernel(ctypes.c_void_p(int(f)),
                                   grid[0], grid[1], grid[2],
                                   block[0], block[1], block[2],
                                   smem, None, kp, None)
        assert r == 0, f"cuLaunchKernel: {r}"
        err, = self.cp.cuCtxSynchronize()
        assert int(err) == 0, f"cuCtxSynchronize: {err}"


# ---------------------------------------------------------------------------
# emulator driver
# ---------------------------------------------------------------------------
def sass_text_of(cubin):
    return subprocess.run([CUOBJDUMP, "-sass", cubin], capture_output=True,
                          text=True, check=True).stdout


def run_case(gpu, case):
    cubin = os.path.join(HERE, case["cubin"])
    func = case["func"]
    grid, block = case["grid"], case["block"]
    smem = case.get("smem", 0)

    # --- lay out host buffers -------------------------------------------
    addr = [0x10000000]
    bufs = {}

    def alloc(nbytes, align=256):
        a = (addr[0] + align - 1) // align * align
        addr[0] = a + max(nbytes, 1)
        return a

    for spec in case["inputs"]:
        data = encode(spec["ty"], spec["values"])
        bufs[spec["name"]] = {"addr": alloc(len(data)), "data": data,
                              "ty": spec["ty"], "n": len(data)}
    for spec in case["outputs"]:
        per = DT[spec["ty"]][0]
        n = spec["n"]
        nbytes = spec.get("nbytes", per * n)
        init = spec.get("init_bytes", b"\x00" * nbytes)
        bufs[spec["name"]] = {"addr": alloc(nbytes), "data": init,
                              "ty": spec["ty"], "n": n, "nbytes": nbytes}

    # --- parameter list, in kernel order --------------------------------
    def param_list(ptr_of):
        out = []
        for p in case["args"]:
            if p[0] == "ptr":
                out.append(("ptr", ptr_of(p[1])))
            else:
                out.append((p[0], p[1]))
        return out

    # --- GPU run ---------------------------------------------------------
    mod = gpu.module(cubin)
    fn = gpu.function(mod, func)
    dev = {}
    for key, b in bufs.items():
        dev[key] = (gpu.memcpy_htod(b["data"]), b["nbytes"] if "nbytes" in b else b["n"])
    gpu.launch(fn, grid, block, param_list(lambda k: dev[k][0]), smem)
    gpu_out = {k: gpu.memcpy_dtoh(dev[k][0], dev[k][1])
               for k in bufs if k in {s["name"] for s in case["outputs"]}}

    # --- emulator run ----------------------------------------------------
    res = sass_emu.resource_usage(cubin)
    static_shared = res.get(func, {}).get("shared", 0)
    m = sass_emu.SASSMachine(sass_text_of(cubin), func=func,
                             max_steps=case.get("max_steps", 60_000_000),
                             static_shared=static_shared)
    m.set_launch(grid, block, smem + static_shared)
    for key, b in bufs.items():
        m.load_bytes(b["addr"], b["data"].ljust(b.get("nbytes", len(b["data"])), b"\x00"))
    m.set_params(param_list(lambda k: bufs[k]["addr"]))
    m.run()
    emu_out = {k: m.read_bytes(bufs[k]["addr"], bufs[k]["nbytes"])
               for k in bufs if "nbytes" in bufs[k]}

    # --- compare ---------------------------------------------------------
    rows = []
    ok = True
    for spec in case["outputs"]:
        key = spec["name"]
        ty, n = spec["ty"], spec["n"]
        g = decode(ty, gpu_out[key], n)
        e = decode(ty, emu_out[key], n)
        byte_eq = gpu_out[key] == emu_out[key]
        ok = ok and byte_eq
        bad = np.nonzero(g != e)[0]
        ref_note = ""
        if "reference" in spec:
            ref = np.array(spec["reference"], dtype=g.dtype)
            ref_note = " ref==gpu" if np.array_equal(ref, g) else " REF!=GPU"
            if not np.array_equal(ref, g):
                ok = False
        rows.append({
            "key": key, "ty": ty, "n": n,
            "byte_eq": byte_eq,
            "nbad": int(len(bad)),
            "first_bad": int(bad[0]) if len(bad) else None,
            "nonzero": int(np.count_nonzero(e)),
            "ref_note": ref_note,
            "gpu": g, "emu": e,
        })
    return {"name": case["name"], "ok": ok, "rows": rows,
            "steps": m.stats["steps"], "warps": m.stats["warps"],
            "barriers": m.stats["barriers"],
            "oob_reads": m.global_space.oob_reads,
            "oob_writes": len(m.global_space.oob_writes),
            "warnings": sorted(set(m.warnings)),
            "unknown": m.stats["unknown"]}


# ---------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------
def case_vec_add(n=1024):
    a = [(i % 97) * 0.5 - 24.0 for i in range(n)]
    b = [float((i * 7) % 41) - 20.0 for i in range(n)]
    return {
        "name": f"vec_add n={n}",
        "cubin": "vec_add.cubin",
        "func": "vec_add",
        "grid": ((n + 255) // 256, 1, 1),
        "block": (256, 1, 1),
        "args": [("ptr", "a"), ("ptr", "b"), ("ptr", "out"), ("i32", n)],
        "inputs": [{"name": "a", "ty": "f32", "values": a},
                   {"name": "b", "ty": "f32", "values": b}],
        "outputs": [{"name": "out", "ty": "f32", "n": n,
                     "reference": [float(np.float32(x) + np.float32(y))
                                   for x, y in zip(a, b)]}],
    }


def case_custom(n=1024):
    a = [(i % 89) * 0.25 - 11.0 for i in range(n)]
    b = [float((i * 3) % 31) * 0.5 - 7.0 for i in range(n)]
    tile = 256
    blocks = (n + tile - 1) // tile
    # The kernel redundantly computes each block's full 256-element dot on
    # EVERY lane (×32 lanes after the shuffle chain), so the reference below
    # is 32× the true dot — model the kernel's behaviour, not the maths.
    ref = []
    for blk in range(blocks):
        acc = np.float32(0.0)
        for i in range(tile):
            g = blk * tile + i
            va = np.float32(a[g]) if g < n else np.float32(0.0)
            vb = np.float32(b[g]) if g < n else np.float32(0.0)
            acc = np.float32(acc + np.float32(va * vb))
        for _ in range(5):  # offsets 16,8,4,2,1
            acc = np.float32(acc + acc)
        ref.append(float(acc))
    return {
        "name": f"custom n={n}",
        "cubin": "custom.cubin",
        "func": "custom_kernel",
        "grid": (blocks, 1, 1),
        "block": (tile, 1, 1),
        "args": [("ptr", "a"), ("ptr", "b"), ("ptr", "out"), ("i32", n)],
        "inputs": [{"name": "a", "ty": "f32", "values": a},
                   {"name": "b", "ty": "f32", "values": b}],
        "outputs": [{"name": "out", "ty": "f32", "n": blocks, "reference": ref}],
    }


def case_moe(numel=100, num_experts=16, use_int64=0, is_ep=0):
    rng = np.random.default_rng(7)
    ids = np.repeat(np.arange(num_experts), 4)
    ids = np.concatenate([ids, rng.integers(0, num_experts, numel - len(ids))])
    ids = np.sort(ids[:numel]).astype("<i4")           # sorted, as the name says
    perm = rng.permutation(numel).astype("<i8")        # shuffled reorder_ids
    threads = 256
    blocks = max(1, (max(numel, num_experts + 1) + threads - 1) // threads)

    neg_count = int(np.searchsorted(ids, 0, side="left"))
    offs = []
    for e in range(num_experts + 1):
        offs.append((int(np.searchsorted(ids, e, side="left")) if e < num_experts
                     else numel) - neg_count)
    s2d = [0] * numel
    for i in range(numel):
        s2d[int(perm[i])] = i - neg_count

    return {
        "name": f"moe numel={numel} exp={num_experts} int64={use_int64}",
        "cubin": "samples/real/moe_permute_prepare.cubin",
        "func": "_ZN6sglang26moe_permute_prepare_kernelEPKiPKlPvPiiibb",
        "grid": (blocks, 1, 1),
        "block": (threads, 1, 1),
        "args": [("ptr", "ids"), ("ptr", "perm"), ("ptr", "offs"),
                 ("ptr", "s2d"), ("i32", num_experts), ("i32", numel),
                 ("i8", use_int64), ("i8", is_ep)],
        "inputs": [{"name": "ids", "ty": "i32", "values": ids.tolist()},
                   {"name": "perm", "ty": "i64", "values": perm.tolist()}],
        "outputs": [
            {"name": "offs", "ty": "i64" if use_int64 else "i32",
             "n": num_experts + 1, "reference": offs},
            {"name": "s2d", "ty": "i32", "n": numel, "reference": s2d},
        ],
    }


def case_transpose(rows=64, cols=48):
    """Tiled transpose with 2-D grid/block and padded shared memory."""
    inp = [float((r * cols + c) % 97) - 48.0 for r in range(rows) for c in range(cols)]
    ref = [0.0] * (rows * cols)
    for r in range(rows):
        for c in range(cols):
            ref[c * rows + r] = inp[r * cols + c]
    tile = 16
    return {
        "name": f"transpose {rows}x{cols}",
        "cubin": "build/transpose.cubin",
        "func": "transpose_kernel",
        "grid": ((cols + tile - 1) // tile, (rows + tile - 1) // tile, 1),
        "block": (tile, tile, 1),
        "args": [("ptr", "in"), ("ptr", "out"), ("i32", rows), ("i32", cols)],
        "inputs": [{"name": "in", "ty": "f32", "values": inp}],
        "outputs": [{"name": "out", "ty": "f32", "n": rows * cols, "reference": ref}],
    }


CASES = {
    "transpose": case_transpose,
    "vec_add": case_vec_add,
    "custom": case_custom,
    "moe": case_moe,
}


def sweep():
    """A spread of sizes/shapes, so a byte-match is not one lucky data point."""
    cases = []
    for n in (256, 1024, 4096):
        cases.append(case_vec_add(n))
    for n in (512, 1024, 2048):
        cases.append(case_custom(n))
    for numel, exps, i64 in ((100, 16, 0), (37, 8, 0), (255, 32, 0),
                             (100, 16, 1), (13, 3, 1), (256, 64, 0)):
        cases.append(case_moe(numel, exps, i64))
    return cases


def main(argv):
    if "--sweep" in argv:
        return run_all(Gpu(), sweep())
    want = argv[1:] or list(CASES)
    gpu = Gpu()
    print(f"{'case':34s} {'buffer':8s} {'n':>6s} {'bytes-equal':>12s} "
          f"{'bad':>5s} {'nonzero':>8s} {'steps':>9s}  ref")
    print("-" * 108)
    allok = True
    for tag in want:
        if tag not in CASES:
            print(f"unknown case {tag}; have {list(CASES)}")
            return 2
        case = CASES[tag]()
        try:
            r = run_case(gpu, case)
        except sass_emu.Unsupported as e:
            print(f"{case['name']:34s} EMULATOR UNSUPPORTED: {e}")
            allok = False
            continue
        allok = allok and r["ok"]
        for row in r["rows"]:
            print(f"{r['name']:34s} {row['key']:8s} {row['n']:>6d} "
                  f"{str(row['byte_eq']):>12s} {row['nbad']:>5d} "
                  f"{row['nonzero']:>8d} {r['steps']:>9d}  {row['ref_note']}")
        extra = []
        if r["barriers"]:
            extra.append(f"barriers={r['barriers']}")
        if r["oob_reads"]:
            extra.append(f"oob_reads={r['oob_reads']}")
        if r["oob_writes"]:
            extra.append(f"OOB_WRITES={r['oob_writes']}")
        if r["unknown"]:
            extra.append(f"unknown_ops={r['unknown']}")
        if r["warnings"]:
            extra.append(f"warnings={r['warnings'][:3]}")
        if extra:
            print(f"{'':34s} {'':8s} {'':>6s}   {' '.join(extra)}")
        if not r["ok"]:
            for row in r["rows"]:
                if not row["byte_eq"]:
                    i = row["first_bad"]
                    print(f"    {row['key']}[{i}] emu={row['emu'][i]} gpu={row['gpu'][i]}")
    return finish(allok)


def finish(allok):
    print("-" * 108)
    print("ALL BYTE-EQUAL" if allok else "MISMATCH")
    return 0 if allok else 1


def run_all(gpu, cases):
    print(f"{'case':34s} {'buffer':8s} {'n':>6s} {'bytes-equal':>12s} "
          f"{'bad':>5s} {'nonzero':>8s} {'steps':>9s}  ref")
    print("-" * 108)
    allok = True
    for case in cases:
        try:
            r = run_case(gpu, case)
        except sass_emu.Unsupported as e:
            print(f"{case['name']:34s} EMULATOR UNSUPPORTED: {e}")
            allok = False
            continue
        allok = allok and r["ok"]
        for row in r["rows"]:
            print(f"{r['name']:34s} {row['key']:8s} {row['n']:>6d} "
                  f"{str(row['byte_eq']):>12s} {row['nbad']:>5d} "
                  f"{row['nonzero']:>8d} {r['steps']:>9d}  {row['ref_note']}")
    return finish(allok)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
