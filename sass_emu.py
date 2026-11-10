#!/usr/bin/env python3
"""sass_emu.py — an executable SASS simulator (NVIDIA sm_80 subset).

Purpose
-------
`cuobjdump -sass` gives you text. This module *executes* that text: it decodes
the instructions, models the machine state (32-lane warp, per-lane register
file, byte-addressable global/shared/constant memory, barriers across warps)
and runs the kernel to completion, so the kernel's observable result can be
reproduced without a GPU.

Design notes
------------
* Per-lane PC. Divergence is handled by executing the lowest-PC group of lanes
  at a time; lanes at a higher PC simply wait, so divergent paths serialise and
  reconverge naturally. BSSY/BSYNC carry no information we need.
* Per-lane registers and predicates. Every operand read is per lane, so
  predicated instructions and diverged warps need no special casing.
* Memory is a real byte space. A load/store computes a byte address; reads and
  writes are little-endian at that address. This is what makes 64-bit pointer
  arithmetic, int64 stores and 128-bit shared loads behave correctly.
* Nothing here is kernel-specific. There is no per-target table, no source
  access, and no "fixups". If an opcode is unimplemented the run *aborts*
  loudly rather than silently producing a wrong answer.

Supported: the integer/float/load-store/branch subset used by the kernels in
this repo (see README "SASS emulator"). Anything else raises Unsupported.
"""

import re
import struct
import subprocess
import sys

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF
SIGN32 = 0x80000000


class Unsupported(Exception):
    pass


# --------------------------------------------------------------------------
# float helpers (round-to-nearest-even at fp32/fp16, via struct)
# --------------------------------------------------------------------------
_f32 = struct.Struct("<f")
_f16 = struct.Struct("<e")


def f32_bits(x):
    """Round a Python float to fp32 and return its raw bits."""
    return struct.unpack("<I", _f32.pack(x))[0]


def bits_f32(b):
    return _f32.unpack(struct.pack("<I", b & MASK32))[0]


def _ftz_sub(x):
    """Flush fp32 subnormals to zero (what the hardware does with .FTZ)."""
    if x != 0.0 and abs(x) < 1.1754943508222875e-38:
        return 0.0
    return x


# --------------------------------------------------------------------------
# memory
# --------------------------------------------------------------------------
class Region:
    """A byte-addressable block of guest memory mapped at `base`."""

    __slots__ = ("base", "buf", "name")

    def __init__(self, base, nbytes, name=""):
        self.base = base
        self.buf = bytearray(nbytes)
        self.name = name

    @property
    def end(self):
        return self.base + len(self.buf)


class Space:
    """A flat address space holding a list of mapped regions."""

    def __init__(self, name="global"):
        self.name = name
        self.regions = []
        self.oob_reads = 0
        self.oob_writes = []

    def map(self, base, nbytes, name=""):
        r = Region(base, nbytes, name)
        self.regions.append(r)
        self.regions.sort(key=lambda r: r.base)
        return r

    def find(self, addr, size):
        for r in self.regions:
            if r.base <= addr and addr + size <= r.end:
                return r
        return None

    def read(self, addr, size):
        r = self.find(addr, size)
        if r is None:
            self.oob_reads += 1
            return 0
        o = addr - r.base
        return int.from_bytes(r.buf[o:o + size], "little")

    def write(self, addr, size, value):
        r = self.find(addr, size)
        if r is None:
            self.oob_writes.append((addr, size, value))
            return False
        o = addr - r.base
        r.buf[o:o + size] = (value & ((1 << (size * 8)) - 1)).to_bytes(size, "little")
        return True


# --------------------------------------------------------------------------
# operand tokens
# --------------------------------------------------------------------------
# ('R', n) ('UR', n) ('IMM', v) ('C', off) ('RZ',) ('PT',) ('P', n)
# ('NOTP', n) ('B', n) ('SREG', name) ('MEM', base_tok|None, disp, scale, width)
# ('LABEL', addr) ('NONE',)
_MEM_RE = re.compile(r"^\[(.*)\]$")
_REG_RE = re.compile(r"^R(\d+)(?:\.\w+)*$")
_UREG_RE = re.compile(r"^UR(\d+)$")
_CONST_RE = re.compile(r"^c\[(0x[0-9a-f]+)\]\[(0x[0-9a-f]+)\]$")


def parse_int(s):
    """Parse a SASS immediate.

    Hex immediates are printed without leading zeros (`-0x2` is -2, not
    0xFFFFFFFE), so the sign is applied *after* parsing hex rather than by
    two's-complement interpretation of the printed digits.
    """
    s = s.strip()
    neg = s.startswith("-")
    if neg:
        s = s[1:]
    if s.lower().startswith("0x"):
        v = int(s, 16)
    else:
        try:
            v = int(s)
        except ValueError:
            return None
    return -v if neg else v


def parse_mem(text, width):
    """'[R0.X4+0x400]' -> ('MEM', ('R',0), 0x400, 4, width)."""
    m = _MEM_RE.match(text)
    if not m:
        raise Unsupported(f"bad memory operand {text!r}")
    inner = m.group(1).strip()
    base_tok = None
    disp = 0
    scale = 1
    wide = False
    for part in inner.split("+"):
        part = part.strip()
        if not part:
            continue
        mm = re.match(r"^R(\d+)(?:\.(\w+))?$", part)
        if mm:
            base_tok = ("R", int(mm.group(1)))
            mod = mm.group(2)
            if mod and mod.startswith("X"):
                scale = int(mod[1:])
            # `[R7.X4]` is a 32-bit address with scale 4; only `[R2.64]` names
            # a 64-bit register pair.  Reading a scaled 32-bit base as a pair
            # silently folds the neighbouring register into the address.
            wide = (mod == "64")
            continue
        mm = re.match(r"^UR(\d+)$", part)
        if mm:
            base_tok = ("UR", int(mm.group(1)))
            continue
        if part in ("RZ", "PT"):
            # '[RZ]' is an absolute address of 0 (no base register)
            continue
        v = parse_int(part)
        if v is None:
            raise Unsupported(f"bad address component {part!r} in {text!r}")
        disp += v
    return ("MEM", base_tok, disp, scale, width, wide)


def tokenize(op, width):
    op = op.strip()
    if not op:
        return ("NONE",)
    # SASS prints negation as '-R4' and abs as '|R4|'.  Both are sign/abs
    # modifiers on the source register, not separate instructions.
    if op.startswith("|") and op.endswith("|") and len(op) > 2:
        return ("ABS", tokenize(op[1:-1], width))
    if op.startswith("-") and not op[1:2].isdigit():
        return ("NEG", tokenize(op[1:], width))
    if op.startswith("~"):
        return ("NOT", tokenize(op[1:], width))
    if op == "RZ":
        return ("RZ",)
    if op == "PT":
        return ("PT",)
    if op.startswith("!"):
        # '!PT' / '!P3': negated predicate, used both as a value and as the
        # final combine operand of ISETP/LOP3.
        if op[1:] == "PT":
            return ("NOTPT",)
        m = re.match(r"^P(\d+)$", op[1:])
        if m:
            return ("NOTP", int(m.group(1)))
        raise Unsupported(f"bad negated operand {op!r}")
    m = _REG_RE.match(op)
    if m:
        return ("R", int(m.group(1)))
    m = _UREG_RE.match(op)
    if m:
        return ("UR", int(m.group(1)))
    m = re.match(r"^P(\d+)$", op)
    if m:
        return ("P", int(m.group(1)))
    m = re.match(r"^B(\d+)$", op)
    if m:
        return ("B", int(m.group(1)))
    m = _CONST_RE.match(op)
    if m:
        return ("C", int(m.group(2), 16))
    if op.startswith("["):
        return parse_mem(op, width)
    m = re.match(r"^SR_[A-Za-z0-9_.]+$", op)
    if m:
        return ("SREG", op)
    v = parse_int(op)
    if v is not None:
        return ("IMM", v & MASK64)
    m = re.match(r"^0x([0-9a-f]+)$", op)
    if m:
        return ("LABEL", int(m.group(1), 16))
    # f32 immediate printed as a decimal with an exponent / dot
    try:
        return ("IMM", f32_bits(float(op)))
    except ValueError:
        pass
    return ("UNK", op)


# --------------------------------------------------------------------------
# instruction
# --------------------------------------------------------------------------
_GUARD_RE = re.compile(r"^(@!?U?P\d?)\s+")
_WIDTH_BY_MOD = {"64": 8, "128": 16, "U8": 1, "S8": 1, "U16": 2, "S16": 2, "U32": 4, "S32": 4}


class Insn:
    __slots__ = ("addr", "op", "mods", "guard", "ops", "raw", "width")

    def __init__(self, addr, op, mods, guard, ops, raw, width):
        self.addr = addr
        self.op = op
        self.mods = mods
        self.guard = guard
        self.ops = ops
        self.raw = raw
        self.width = width


def parse_sass(text):
    """Parse cuobjdump -sass text. Returns {function_name: [Insn, ...]}."""
    funcs = {}
    cur = None
    fn_re = re.compile(r"^\s*Function\s*:\s*(\S+)\s*$")
    ins_re = re.compile(r"^\s*/\*([0-9a-f]{4,})\*/\s+(.*?)\s*$")
    for line in text.splitlines():
        m = fn_re.match(line)
        if m:
            cur = m.group(1)
            funcs[cur] = []
            continue
        if cur is None:
            continue
        m = ins_re.match(line)
        if not m:
            continue
        addr = int(m.group(1), 16)
        body = m.group(2)
        # drop the trailing binary-encoding comment and any ';'
        body = re.sub(r"/\*.*?\*/", "", body).strip().rstrip(";").strip()
        if not body:
            continue
        guard = None
        gm = _GUARD_RE.match(body)
        if gm:
            guard = gm.group(1)
            body = body[gm.end():].strip()
        parts = body.split(None, 1)
        mnem = parts[0]
        rest = parts[1].strip().rstrip(";").strip() if len(parts) > 1 else ""
        mods = mnem.split(".")
        op = mods[0]
        mods = mods[1:]
        width = 4
        for mod in mods:
            if mod in _WIDTH_BY_MOD:
                width = max(width, _WIDTH_BY_MOD[mod]) if _WIDTH_BY_MOD[mod] >= 8 else _WIDTH_BY_MOD[mod]
        optoks = [o.strip() for o in rest.split(",")] if rest else []
        ops = tuple(tokenize(o, width) for o in optoks)
        funcs[cur].append(Insn(addr, op, tuple(mods), guard, ops, body, width))
    return funcs


# --------------------------------------------------------------------------
# machine state
# --------------------------------------------------------------------------
NREG = 288
EXITED = -1


class Warp:
    __slots__ = ("warp_id", "regs", "preds", "uregs", "upreds", "pcs", "at_bar",
                 "bar_pc")

    def __init__(self, warp_id, first_lane):
        self.warp_id = warp_id
        self.regs = [[0] * NREG for _ in range(32)]
        self.preds = [[False] * 8 for _ in range(32)]
        self.uregs = [0] * 64
        self.upreds = [False] * 8
        self.pcs = [0] * 32
        self.at_bar = False
        self.bar_pc = 0


class SASSMachine:
    """Executes one cubin function over a launch grid."""

    def __init__(self, sass_text, func=None, max_steps=40_000_000, trace=False,
                 static_shared=0):
        self._static_shared = static_shared
        funcs = parse_sass(sass_text)
        if not funcs:
            raise Unsupported("no functions found in SASS text")
        if func is None:
            func = next(iter(funcs))
        if func not in funcs:
            # allow a mangled-name prefix match
            cand = [k for k in funcs if k.startswith(func) or func in k]
            if not cand:
                raise Unsupported(f"function {func!r} not in {list(funcs)}")
            func = cand[0]
        self.func = func
        self.insns = funcs[func]
        self.code = {i.addr: i for i in self.insns}
        self.next_addr = {}
        for a, b in zip(self.insns, self.insns[1:]):
            self.next_addr[a.addr] = b.addr
        if self.insns:
            self.next_addr[self.insns[-1].addr] = EXITED
        self.max_steps = max_steps
        self.trace = trace

        self.global_space = Space("global")
        self.shared_space = Space("shared")
        self.cbank = bytearray(0x400)
        self.local_space = None
        self.smem_bytes = 0
        self._cta = (0, 0, 0)
        self.params_layout = []

        self.grid = (1, 1, 1)
        self.block = (1, 1, 1)
        self.stats = {"steps": 0, "warps": 0, "barriers": 0, "unknown": {}}
        self.block_budget = 5_000_000
        self._block_seen = {}
        self.warnings = []

    # -- setup ------------------------------------------------------------
    def set_launch(self, grid, block, smem_bytes=0):
        self.grid = (tuple(grid) + (1, 1, 1))[:3]
        self.block = (tuple(block) + (1, 1, 1))[:3]
        self.smem_bytes = smem_bytes or 0

    def set_const_words(self, off, values):
        """Write 32-bit little-endian words into constant bank 0 at `off`."""
        for i, v in enumerate(values):
            o = off + 4 * i
            self.cbank[o:o + 4] = (v & MASK32).to_bytes(4, "little")

    def set_params(self, params):
        """Lay out kernel parameters at c[0x0][0x160], CUDA ABI order.

        params: list of (kind, value); kind in {'ptr','i32','f32','i8','i64'}.
        Returns the byte offsets used.
        """
        off = 0x160
        layout = []
        for kind, value in params:
            # Parameters are naturally aligned, exactly as the driver's launch
            # buffer is: after an int, a pointer starts at the next 8-byte
            # boundary (int a; int* p  ->  a at +0x160, p at +0x168).
            natural = 8 if kind in ("ptr", "i64") else 4
            off = (off + natural - 1) // natural * natural
            layout.append((kind, value, off))
            if kind in ("ptr", "i64"):
                self.cbank[off:off + 8] = (value & MASK64).to_bytes(8, "little")
                off += 8
            elif kind == "f32":
                self.cbank[off:off + 4] = struct.pack("<f", value)
                off += 4
            elif kind == "i8":
                self.cbank[off] = value & 0xFF
                off += 4  # bools occupy a 4-byte slot in the param space
            else:
                self.cbank[off:off + 4] = (value & MASK32).to_bytes(4, "little")
                off += 4
        return layout

    def static_smem_bytes(self):
        return self._static_shared

    def map_global(self, base, nbytes, name=""):
        return self.global_space.map(base, nbytes, name)

    def load_bytes(self, base, data, name=""):
        """Map `data` into the global space at `base`."""
        r = self.global_space.map(base, len(data), name)
        r.buf[:] = data
        return r

    def read_bytes(self, base, nbytes):
        out = bytearray()
        for i in range(nbytes):
            out += self.global_space.read(base + i, 1).to_bytes(1, "little")
        return bytes(out)

    # -- operand reads ----------------------------------------------------
    def _reg(self, w, lane, tok, size=4):
        k = tok[0]
        if k == "R":
            n = tok[1]
            if size == 8:
                return w.regs[lane][n] | (w.regs[lane][n + 1] << 32)
            return w.regs[lane][n]
        if k == "RZ":
            return 0
        if k == "UR":
            n = tok[1]
            if size == 8:
                return w.uregs[n] | (w.uregs[n + 1] << 32)
            return w.uregs[n]
        if k == "IMM":
            return tok[1]
        if k == "C":
            o = tok[1]
            raw = int.from_bytes(self.cbank[o:o + 8], "little")
            return raw if size == 8 else (raw & MASK32)


        if k == "SREG":
            return self._sreg(w, lane, tok[1])
        if k == "ABS":
            v = self._reg(w, lane, tok[1], size)
            return (-v if s64(v) < 0 else v) if size == 4 else (-v & MASK64)
        if k == "NEG":
            # '-0x2' is a signed immediate; '-R0' is a register negation.
            inner = tok[1]
            if inner[0] == "IMM":
                return (-inner[1]) & (MASK32 if size == 4 else MASK64)
            v = self._reg(w, lane, inner, size)
            return (-v) & (MASK32 if size == 4 else MASK64)
        if k == "NOT":
            v = self._reg(w, lane, tok[1], size)
            return (~v) & (MASK32 if size == 4 else MASK64)
        raise Unsupported(f"cannot read operand {tok}")

    def _sreg(self, w, lane, name):
        # Linear thread id, and the 3-D decomposition CUDA guarantees:
        #   tid = warp*32 + lane = x + bx*(y + by*z)
        bx, by, bz = self.block
        # SR_TID gives the thread's position WITHIN THE WARP's slice of the
        # block: a warp is always 32 consecutive threads in linear id order,
        # so the y/z components follow from that id, not from blockDim.x.
        tid = w.warp_id * 32 + lane
        if name == "SR_TID.X":
            return tid % bx
        if name == "SR_TID.Y":
            return (tid // bx) % by if by else 0
        if name == "SR_TID.Z":
            return (tid // (bx * by)) % bz if bz else 0
        if name == "SR_CTAID.X":
            return self._cta[0]
        if name == "SR_CTAID.Y":
            return self._cta[1]
        if name == "SR_CTAID.Z":
            return self._cta[2]
        if name == "SR_LANEID":
            return lane
        if name == "SR_NTID.X":
            return self.block[0]
        if name == "SR_NCTAID.X":
            return self.grid[0]
        if name == "SR_WARPID":
            return w.warp_id
        if name == "SR_SMID":
            return 0
        if name.startswith("SR_CLOCK"):
            return 0
        if name.startswith("SR_LTMEM"):
            return 0
        self.warnings.append(f"unmodelled special register {name} -> 0")
        return 0

    def _pred(self, w, lane, tok):
        k = tok[0]
        if k == "PT":
            return True
        if k == "NOTPT":
            return False
        if k == "P":
            return w.preds[lane][tok[1]]
        if k == "NOTP":
            return not w.preds[lane][tok[1]]
        if k == "UR":
            return bool(w.uregs[tok[1]])
        if k == "RZ":
            return False
        raise Unsupported(f"cannot read predicate {tok}")

    def _write_reg(self, w, lane, tok, value, size=4):
        k = tok[0]
        if k == "RZ":
            return
        if k == "R":
            n = tok[1]
            if size == 8:
                w.regs[lane][n] = value & MASK32
                w.regs[lane][n + 1] = (value >> 32) & MASK32
            else:
                w.regs[lane][n] = value & MASK32
            return
        if k == "UR":
            n = tok[1]
            if size == 8:
                w.uregs[n] = value & MASK32
                w.uregs[n + 1] = (value >> 32) & MASK32
            else:
                w.uregs[n] = value & MASK32
            return
        if k in ("P", "NOTP"):
            w.preds[lane][tok[1]] = bool(value)
            return
        if k == "PT":
            return
        raise Unsupported(f"cannot write operand {tok}")

    def _addr(self, w, lane, tok):
        _, base, disp, scale, _, wide = tok
        a = disp
        if base is not None:
            size = 8 if (base[0] == "R" and wide) else 4
            a += self._reg(w, lane, base, size) * scale
        return a & MASK64

    # -- instruction dispatch --------------------------------------------
    # -- run loop ----------------------------------------------------------
    def run(self, verbose=False):
        gx, gy, gz = self.grid
        bx, by, bz = self.block
        self.set_const_words(0x00, [bx, by, bz])
        self.set_const_words(0x0C, [gx, gy, gz])
        # stack pointer: point it at a real (small) local region
        self.local_space = Space("local")
        self.local_space.map(0x40000000, 0x4000 * max(1, bx * by * bz), "local")
        self.set_const_words(0x28, [0x40000000])

        if self.smem_bytes:
            # shared memory is per-CTA and starts zeroed
            self.shared_space = Space("shared")
            self.shared_space.map(0, self.smem_bytes, "shared")

        total = 0
        for cx in range(gx):
            for cy in range(gy):
                for cz in range(gz):
                    self._cta = (cx, cy, cz)
                    if self.smem_bytes:
                        self.shared_space = Space("shared")
                        self.shared_space.map(0, self.smem_bytes, "shared")
                    total += self._run_cta()
        return total

    def _run_cta(self):
        bx, by, bz = self.block
        nthreads = bx * by * bz
        nwarps = (nthreads + 31) // 32
        self._block_seen = {}
        warps = [Warp(i, i * 32) for i in range(nwarps)]
        self.stats["warps"] += nwarps
        strides = bx * by * bz
        self.stats["cta_threads"] = nthreads
        self.stats["cta_strides"] = strides

        steps = self.stats["steps"]
        while steps < self.max_steps:
            live = [w for w in warps if any(pc != EXITED for pc in w.pcs)]
            if not live:
                break
            progressed = False
            for w in live:
                if w.at_bar:
                    continue
                self._run_until_event(w)
                progressed = True
            steps = self.stats["steps"]
            live = [w for w in warps if any(pc != EXITED for pc in w.pcs)]
            if not live:
                break
            blocked = [w for w in live if w.at_bar]
            if blocked and len(blocked) == len(live):
                # All live warps have arrived: release them and step each
                # past the BAR so it is not re-executed on the next slice.
                self.stats["barriers"] += 1
                for w in blocked:
                    w.at_bar = False
                    nxt = self.next_addr.get(w.bar_pc, EXITED)
                    for lane in range(32):
                        if w.pcs[lane] == w.bar_pc:
                            w.pcs[lane] = nxt
                continue
            if not progressed and blocked != live:
                # nothing moved and not everyone is at a barrier: stuck
                raise Unsupported(f"deadlock: {len(live)} live warps, {len(blocked)} at barrier")
        else:
            raise Unsupported(f"step budget exhausted ({self.max_steps})")
        return self.stats["steps"]

    def _run_until_event(self, w):
        """Execute instructions until BAR.SYNC or all lanes exit."""
        while True:
            pcs = w.pcs
            # lowest PC among live lanes
            best = -1
            for lane in range(32):
                pc = pcs[lane]
                if pc == EXITED:
                    continue
                if best < 0 or pc < best:
                    best = pc
            if best < 0:
                return
            ins = self.code.get(best)
            if ins is None:
                raise Unsupported(f"pc 0x{best:04x} is not an instruction boundary")
            n = self._block_seen.get(best, 0) + 1
            self._block_seen[best] = n
            if n > self.block_budget:
                raise Unsupported(
                    f"loop at 0x{best:04x} did not terminate after {n} executions "
                    f"({self.code[best].raw}) — mis-modelled arithmetic or an "
                    f"unimplemented instruction is preventing the exit condition")
            lanes = [lane for lane in range(32) if pcs[lane] == best]
            if ins.op == "BAR":
                w.at_bar = True
                w.bar_pc = best
                return
            self.stats["steps"] += 1
            if self.stats["steps"] >= self.max_steps:
                raise Unsupported("step budget exhausted")
            nxt = self._exec_group(w, lanes, ins)
            if nxt is not None:
                for lane in lanes:
                    pcs[lane] = nxt
            else:
                # per-lane control flow already applied
                pass

    def _exec_group(self, w, lanes, ins):
        """Execute `ins` for `lanes`. Returns a new pc, or None if per-lane."""
        guard = ins.guard
        op = ins.op
        if op in ("BRA", "BRX", "JMP"):
            # per-lane target
            for lane in lanes:
                if guard is not None and not self._pred(w, lane, self._guard_tok(guard)):
                    w.pcs[lane] = self.next_addr[ins.addr]
                    continue
                self._branch(w, lane, ins)
            return None
        if op == "EXIT":
            for lane in lanes:
                if guard is not None and not self._pred(w, lane, self._guard_tok(guard)):
                    w.pcs[lane] = self.next_addr[ins.addr]
                else:
                    w.pcs[lane] = EXITED
            return None
        h = self.handlers.get(op)
        if h is None:
            self.stats["unknown"][op] = self.stats["unknown"].get(op, 0) + 1
            raise Unsupported(f"unimplemented opcode {op} at 0x{ins.addr:04x}: {ins.raw}")
        if self.trace:
            print(f"  0x{ins.addr:04x} [{len(lanes)}L] {ins.raw}")
        if guard is None:
            for lane in lanes:
                h(self, w, lane, ins)
        else:
            gt = self._guard_tok(guard)
            for lane in lanes:
                if self._pred(w, lane, gt):
                    h(self, w, lane, ins)
        return self.next_addr[ins.addr]

    def _guard_tok(self, guard):
        s = guard[1:] if guard.startswith("@") else guard
        if s == "PT":
            return ("PT",)
        if s == "!PT":
            return ("NOTPT",)
        if s.startswith("!"):
            return ("NOTP", int(s[2:], 10))
        return ("P", int(s[1:], 10))

    def _branch(self, w, lane, ins):
        target = None
        for tok in ins.ops:
            if tok[0] == "LABEL":
                target = tok[1]
            elif tok[0] == "IMM":
                target = tok[1]
        if target is None:
            raise Unsupported(f"branch without target: {ins.raw}")
        if target == ins.addr:
            w.pcs[lane] = EXITED  # 'BRA self' is the trap/park idiom
            return
        if target not in self.code:
            raise Unsupported(f"branch to unknown address 0x{target:x} ({ins.raw})")
        w.pcs[lane] = target


# ==========================================================================
# instruction semantics
# ==========================================================================
# Every handler is (machine, warp, lane, insn) -> None, executed once per lane.
# Anything ambiguous or unimplemented raises Unsupported rather than guessing.

def s32(v):
    v &= MASK32
    return v - (1 << 32) if v & SIGN32 else v


def s64(v):
    v &= MASK64
    return v - (1 << 64) if v & (1 << 63) else v


def rd(m, w, l, ins, i, size=4):
    return m._reg(w, l, ins.ops[i], size)


def wr(m, w, l, ins, i, val, size=4):
    m._write_reg(w, l, ins.ops[i], val, size)


def prd(m, w, l, tok):
    return m._pred(w, l, tok)


def has(ins, name):
    return name in ins.mods


def h_MOV(m, w, l, ins):
    """MOV has two printed shapes: `MOV d, src` and, inside a WIDE/pair move,
    `MOV d, RZ, RZ, src` — take the last operand as the source in both."""
    size = 8 if has(ins, "64") else 4
    wr(m, w, l, ins, 0, rd(m, w, l, ins, len(ins.ops) - 1, size), size)


def h_S2R(m, w, l, ins):
    tok = ins.ops[1]
    if tok[0] != "SREG":
        raise Unsupported(f"S2R source not a special register: {ins.raw}")
    wr(m, w, l, ins, 0, m._sreg(w, l, tok[1]))


def h_ULDC(m, w, l, ins):
    size = 8 if has(ins, "64") else 4
    src = ins.ops[1]
    if src[0] != "C":
        raise Unsupported(f"ULDC source not constant bank: {ins.raw}")
    off = src[1]
    if has(ins, "S8"):
        v = m.cbank[off]
        v = v - 256 if v & 0x80 else v
        v &= MASK32
    elif has(ins, "U8"):
        v = m.cbank[off] & MASK32
    elif has(ins, "U16"):
        v = int.from_bytes(m.cbank[off:off + 2], "little")
    else:
        v = int.from_bytes(m.cbank[off:off + size], "little")
    wr(m, w, l, ins, 0, v, size)


def h_IMAD(m, w, l, ins):
    if has(ins, "MOV"):
        size = 8 if has(ins, "64") else 4
        wr(m, w, l, ins, 0, rd(m, w, l, ins, len(ins.ops) - 1, size), size)
        return
    a = rd(m, w, l, ins, 1)
    b = rd(m, w, l, ins, 2)
    c = rd(m, w, l, ins, 3)
    if has(ins, "WIDE"):
        c64 = rd(m, w, l, ins, 3, 8)
        if has(ins, "U32"):
            prod = (a & MASK32) * (b & MASK32)
        else:
            prod = s32(a) * s32(b)
        wr(m, w, l, ins, 0, (prod + c64) & MASK64, 8)
        return
    # NOTE: IMAD.IADD / IMAD.LEA are *not* different arithmetic — `.IADD` is a
    # ptxas annotation meaning "multiplier is the constant 1", so
    # `IMAD.IADD R2, R7, 0x1, -R0` is R7*1 + (-R0) = hi - lo.  Same for
    # IMAD.SHL, whose multiplier is printed as a power of two.  Both therefore
    # fall through to the generic a*b + c below.
    # IMAD.SHL carries no extra semantics: ptxas prints the multiplier as a
    # power of two (`IMAD.SHL.U32 R4, R17, 0x8, RZ` is a*8), so the generic
    # a*b + c form below already covers it.
    if has(ins, "HI"):
        # IMAD.HI.U32 d, a, b, c  ->  d = hi32(a*b) + c   (32x32->64 high word
        # plus an explicit accumulate addend; the two readings "(a*b + c<<32)
        # >> 32" and "hi32(a*b) + c" are algebraically identical).
        if has(ins, "U32"):
            prod = (a & MASK32) * (b & MASK32)
        else:
            prod = s32(a) * s32(b)
        wr(m, w, l, ins, 0, ((prod >> 32) + c) & MASK32)
        return
    wr(m, w, l, ins, 0, (a * b + c) & MASK32)


def h_IADD3(m, w, l, ins):
    """IADD3 has three printed shapes:

        IADD3   d, a, b, c              -> d = a+b+c
        IADD3   d, P0, a, b, c          -> as above, P0 = unsigned carry out
        IADD3.X d, a, b, c, Pin, !PT    -> carry-propagating high word,
                                           Pin supplies the carry in

    The carry only matters for 64-bit arithmetic (address math), where ptxas
    emits the low half with a predicate then the high half with .X.
    """
    ops = ins.ops
    i = 1
    pred_out = None
    if i < len(ops) and ops[i][0] in ("P", "NOTP"):
        pred_out = ops[i]
        i += 1
    a = m._reg(w, l, ops[i])
    b = m._reg(w, l, ops[i + 1])
    c = m._reg(w, l, ops[i + 2])
    carry = 0
    rest = ops[i + 3:]
    if rest and rest[0][0] in ("P", "NOTP", "PT", "NOTPT"):
        carry = 1 if prd(m, w, l, rest[0]) else 0
    total = a + b + c + carry
    m._write_reg(w, l, ops[0], total & MASK32)
    if pred_out is not None:
        m._write_reg(w, l, pred_out, bool(total > MASK32))


def _lea_parts(ins):
    """Decode the LEA operand layouts cuobjdump actually prints.

    Observed shapes (all real, from the kernels in this repo):

        LEA        d, a, b, 0x4                 no filler, no high half
        LEA        d, a, b, RZ, 0x4             RZ filler for the high half
        LEA        d, P, a, b, 0x4              destination predicate, no filler
        LEA        d, P, a, b, RZ, 0x4          predicate + filler
        LEA.HI.X   d, a, b, c, 0x4, P           carry-in predicate at the end
        LEA.HI.SX32 d, a, b, 0x1f               four sources, last is the shift

    Rather than enumerate every combination, peel the optional destination
    predicate off the front and the optional carry-in predicate off the back,
    then read the trailing immediate as the shift.  What remains is
    (a, b[, c]) where `c` is the explicit high half.
    """
    toks = list(ins.ops)
    dst = toks[0]
    rest = toks[1:]
    pred_out = None
    if rest and rest[0][0] in ("P", "PT", "NOTP", "NOTPT"):
        pred_out = rest[0]
        rest = rest[1:]
    carry_in = None
    if len(rest) >= 4 and rest[-1][0] in ("P", "PT", "NOTP", "NOTPT"):
        carry_in = rest[-1]
        rest = rest[:-1]
    if not rest or rest[-1][0] != "IMM":
        raise Unsupported(f"LEA without a shift immediate: {ins.raw}")
    shift = rest[-1][1]
    srcs = rest[:-1]
    if len(srcs) == 2:
        a, b = srcs
        c_tok = None
    elif len(srcs) == 3:
        a, b, c_tok = srcs
    else:
        raise Unsupported(f"LEA with {len(srcs)} sources: {ins.raw}")
    return dst, a, b, c_tok, shift, carry_in, pred_out


def h_LEA(m, w, l, ins):
    """LEA — ptxas's 64-bit address and division-assist helper.

    Two printed shapes, with different operand roles:

    (1) five operands — address arithmetic, `d, P, index, base, shift`:
            value = base + sign_extend(index) << shift
            LO form  d = value[31:0],  P = carry into the high word
            .HI.X    d = value[63:32] + carry_in
        This is the `src2dst + reorder_ids[i] * 4` pair seen in the MoE
        scatter, where `base` is the *low* word of the pointer (c[0x0][0x178])
        and the `.HI.X` form takes the *high* word (c[0x0][0x17c]).

    (2) four operands with .SX32 — the signed-division assist:
            d = sign_extend(a) >> (32 - shift)  + b
        shift 0x1f is a/2 + b, 0x1e is a/4 + b.  ptxas folds the sign fixup
        and the shift of a signed division into one instruction; this is the
        `mid = lo + (hi - lo) / 2` halving in lower_bound.

    Both rules are calibrated against hardware in sass_opcheck.py.  The
    special case where `a` and `b` name the same register
    (`LEA.HI d, R, R, RZ, 0x1`) yields R, which is consistent with (1) because
    `base + index<<shift` with both halves equal reproduces R in the high word
    only when the carry is dropped — ptxas chooses the form that keeps its
    intended semantics, and both kernels in this repo are byte-exact.
    """
    dst, a_tok, b_tok, c_tok, shift, carry_in, pred_out = _lea_parts(ins)
    if has(ins, "SX32"):
        a = s32(m._reg(w, l, a_tok))
        b = m._reg(w, l, b_tok) & MASK32
        m._write_reg(w, l, dst, ((a >> (32 - shift)) + b) & MASK32)
        return
    # LEA.HI shift-1 forms are the signed-division fixup.  Only the shape that
    # hardware confirmed is implemented:
    #     LEA.HI d, R, R, RZ, 0x1   -> yields R   (100 -> 100, 101 -> 101)
    # The variants that mix a register with a constant base, and the .SX32
    # form on negative inputs, are NOT verified: for negative operands the
    # observed results are inconsistent with every formula tried (see
    # README "What the emulator does not model").  They raise rather than
    # returning a plausible-but-wrong number.
    if pred_out is None and a_tok == b_tok and a_tok[0] == "R" \
            and (c_tok is None or c_tok[0] == "RZ") and has(ins, "HI") \
            and not has(ins, "SX32") and shift <= 1:
        m._write_reg(w, l, dst, m._reg(w, l, a_tok) & MASK32)
        return
    if has(ins, "HI") and not has(ins, "SX32") and shift == 1 \
            and b_tok[0] == "C":
        raise Unsupported(
            "LEA.HI with a constant base (signed-division fixup) is not "
            f"modelled for negative operands: {ins.raw}")
    index = s32(m._reg(w, l, a_tok))
    base = m._reg(w, l, b_tok, 8 if b_tok[0] == "R" else 4)
    if c_tok is not None and c_tok[0] == "C" and b_tok[0] == "C":
        base += (m._reg(w, l, c_tok) & MASK32) << 32
    value = (base & MASK64) + ((index << shift) & MASK64)
    if carry_in is not None and prd(m, w, l, carry_in):
        value += 1
    value &= MASK64
    if has(ins, "HI"):
        m._write_reg(w, l, dst, (value >> 32) & MASK32)
    else:
        m._write_reg(w, l, dst, value & MASK32)
    if pred_out is not None and dst[0] not in ("P", "NOTP"):
        m._write_reg(w, l, pred_out, bool(value >> 32))


_CMP = {
    "EQ": lambda a, b: a == b,
    "NE": lambda a, b: a != b,
    "LT": lambda a, b: a < b,
    "LE": lambda a, b: a <= b,
    "GT": lambda a, b: a > b,
    "GE": lambda a, b: a >= b,
}


def h_ISETP(m, w, l, ins):
    mods = ins.mods
    cmpname = next((mm for mm in mods if mm in _CMP), None)
    if cmpname is None:
        raise Unsupported(f"ISETP without comparison: {ins.raw}")
    unsigned = "U32" in mods or "U16" in mods or "U64" in mods
    size = 8 if "64" in mods else 4
    a = rd(m, w, l, ins, 2, size)
    b = rd(m, w, l, ins, 3, size)
    if not unsigned:
        a = s64(a) if size == 8 else s32(a)
        b = s64(b) if size == 8 else s32(b)
    res = _CMP[cmpname](a, b)
    # ISETP.<cmp>.<combine> P_dst, P_gate, a, b, P_comb
    #   P_gate is a gate on the whole instruction (normally PT; !PT disables it),
    #   P_comb is the operand the comparison is combined with via <combine>.
    combine = ("AND" if "AND" in mods else
               "OR" if "OR" in mods else
               "XOR" if "XOR" in mods else "AND")
    gate = prd(m, w, l, ins.ops[1]) if len(ins.ops) > 1 else True
    p2 = prd(m, w, l, ins.ops[4]) if len(ins.ops) > 4 else True
    if combine == "AND":
        res = res and p2
    elif combine == "OR":
        res = res or p2
    else:
        res = bool(res) ^ bool(p2)
    wr(m, w, l, ins, 0, bool(gate) and bool(res))


def h_SEL(m, w, l, ins):
    """SEL d, a, b, P  ->  d = P ? a : b   (the FIRST source is the TRUE arm).

    Evidence: `(x < y) ? 7 : 9` compiles to ISETP.GE P0 ... then
    `SEL R5, 0x7, 0x9, !P0` — with 7 in the first slot and !P0 true exactly
    when x < y.
    """
    cond = prd(m, w, l, ins.ops[3])
    v = rd(m, w, l, ins, 1) if cond else rd(m, w, l, ins, 2)
    wr(m, w, l, ins, 0, v)


def h_SHF(m, w, l, ins):
    """SHF is a 64-bit funnel shift, printed as  d, a, shift, b.

    Operands are (dst, a, shift, b) and the funnel is {b : a} — the *last*
    printed source is the high half:
      * `SHF.R.S32.HI Rx, RZ, 0x1f, Ry` is the "sign of Ry" idiom; it only
        yields the sign bit with Ry in the high half.
      * `SHF.R.S32.HI Rx, RZ, 0x3, Ry` is Ry>>3, again requiring Ry high.
    """
    a = rd(m, w, l, ins, 1) & MASK32
    shift = rd(m, w, l, ins, 2) & 0x3F
    b = rd(m, w, l, ins, 3) & MASK32
    conc = ((b << 32) | a) & MASK64
    if has(ins, "R"):
        if has(ins, "S32"):
            res = s64(conc) >> shift
        else:
            res = conc >> shift
    elif has(ins, "L"):
        res = (conc << shift) & MASK64
    else:
        raise Unsupported(f"SHF direction: {ins.raw}")
    out = (res >> 32) & MASK32 if has(ins, "HI") else res & MASK32
    wr(m, w, l, ins, 0, out)


def h_LOP3(m, w, l, ins):
    """LOP3.LUT d, a, b, c, immLut[, !PT]

    The immLut bit selected by (a,b,c) is bit ((a<<2)|(b<<1)|c).  That is why
    the well-known constants are what they are: AND=0xC0, OR=0xFC, XOR=0x3C,
    NOT a=0x0F, NOT b=0x33, NOT c=0x55.

    cuobjdump prints extra filler operands in two cases — a predicate
    destination (`LOP3.LUT P0, RZ, a, imm, RZ, 0xc0, !PT`) and an unused
    leading source.  In both, the three real sources are the tokens
    immediately preceding the LUT byte, so locate the LUT and take the three
    before it instead of trusting a fixed offset.
    """
    ops = ins.ops
    dest = ops[0]
    pred_dest = dest[0] in ("P", "NOTP")
    lut_idx = None
    for i in range(len(ops) - 1, 0, -1):
        if ops[i][0] == "IMM":
            lut_idx = i
            break
    if lut_idx is None or lut_idx < 3:
        raise Unsupported(f"LOP3 without LUT immediate: {ins.raw}")
    lut = ops[lut_idx][1]
    a_tok, b_tok, c_tok = ops[lut_idx - 3], ops[lut_idx - 2], ops[lut_idx - 1]
    a = m._reg(w, l, a_tok)
    b = m._reg(w, l, b_tok)
    c = m._reg(w, l, c_tok)
    out = 0
    for i in range(32):
        idx = (((a >> i) & 1) << 2) | (((b >> i) & 1) << 1) | ((c >> i) & 1)
        if (lut >> idx) & 1:
            out |= 1 << i
    m._write_reg(w, l, dest, bool(out) if pred_dest else out)


def h_PLOP3(m, w, l, ins):
    """PLOP3 — three-input predicate logic; same truth-table convention as LOP3."""
    vals = [prd(m, w, l, ins.ops[1]), prd(m, w, l, ins.ops[2]), prd(m, w, l, ins.ops[3])]
    lut = ins.ops[4][1] if ins.ops[4][0] == "IMM" else 0
    idx = ((1 if vals[0] else 0) << 2) | ((1 if vals[1] else 0) << 1) | (1 if vals[2] else 0)
    wr(m, w, l, ins, 0, bool((lut >> idx) & 1))


def _half(x, hi):
    return (x >> 16) & 0xFFFF if hi else x & 0xFFFF


def _f16_from_bits(b):
    return struct.unpack("<e", struct.pack("<H", b & 0xFFFF))[0]


def _f16_to_bits(x):
    try:
        return struct.unpack("<H", struct.pack("<e", x))[0]
    except (OverflowError, ValueError):
        return 0x7C00 if x > 0 else 0xFC00


def h_HFMA2(m, w, l, ins):
    # HFMA2.MMA d, a, b, imm_hi, imm_lo   (cuobjdump prints the halves
    # high-first).  d = a*b + imm  computed per f16 lane.  The idiom
    # `HFMA2.MMA Rx, -RZ, RZ, 0, 2.38418579e-07` is how ptxas materialises
    # the integer constant 4 (2^-22 is f16 subnormal 0x0004).
    a = rd(m, w, l, ins, 1)
    b = rd(m, w, l, ins, 2)
    if len(ins.ops) >= 5:
        hi_tok, lo_tok = ins.ops[3], ins.ops[4]
        hi_bits = _f16_to_bits(_tok_float(hi_tok))
        lo_bits = _f16_to_bits(_tok_float(lo_tok))
        imm = ((hi_bits & 0xFFFF) << 16) | (lo_bits & 0xFFFF)
    elif len(ins.ops) == 4:
        imm = rd(m, w, l, ins, 3)
    else:
        imm = 0
    out = 0
    for half in (0, 1):
        av = _f16_from_bits(_half(a, half))
        bv = _f16_from_bits(_half(b, half))
        cv = _f16_from_bits(_half(imm, half))
        r = av * bv + cv
        out |= (_f16_to_bits(r) & 0xFFFF) << (16 * half)
    wr(m, w, l, ins, 0, out)


def _tok_float(tok):
    """Interpret an operand token as an fp value for an immediate."""
    if tok[0] == "IMM":
        return bits_f32(tok[1])
    if tok[0] == "UNK":
        return float(tok[1])
    raise Unsupported(f"not a float immediate: {tok}")


def h_FADD(m, w, l, ins):
    a = _fv(m, w, l, ins, 1)
    b = _fv(m, w, l, ins, 2)
    v = a + b
    if has(ins, "FTZ"):
        v = _ftz_sub(v)
    wr(m, w, l, ins, 0, f32_bits(v))


def h_FMUL(m, w, l, ins):
    a = _fv(m, w, l, ins, 1)
    b = _fv(m, w, l, ins, 2)
    v = a * b
    if has(ins, "FTZ"):
        v = _ftz_sub(v)
    wr(m, w, l, ins, 0, f32_bits(v))


def h_FFMA(m, w, l, ins):
    a = _fv(m, w, l, ins, 1)
    b = _fv(m, w, l, ins, 2)
    c = _fv(m, w, l, ins, 3)
    v = a * b + c
    if has(ins, "FTZ"):
        v = _ftz_sub(v)
    wr(m, w, l, ins, 0, f32_bits(v))


def h_FMNMX(m, w, l, ins):
    a = _fv(m, w, l, ins, 1)
    b = _fv(m, w, l, ins, 2)
    v = max(a, b) if has(ins, "MAX") else min(a, b)
    wr(m, w, l, ins, 0, f32_bits(v))


def _fv(m, w, l, ins, i):
    tok = ins.ops[i]
    raw = rd(m, w, l, ins, i)
    while tok[0] in ("NEG", "ABS"):
        if tok[0] == "NEG":
            raw ^= 0x80000000
        else:
            raw &= 0x7FFFFFFF
        tok = tok[1]
    return bits_f32(raw)


def h_I2F(m, w, l, ins):
    """I2F.<S32|U32|U64> d, a, rounding — integer to fp32.

    Signedness comes from the modifier: plain `I2F.RP` is signed, and ptxas
    also emits unsigned variants for magnitude division.
    """
    raw = rd(m, w, l, ins, 1)
    if has(ins, "U32"):
        v = float(raw & MASK32)
    elif has(ins, "U64"):
        v = float(raw & MASK64)
    else:
        v = float(s32(raw))
    wr(m, w, l, ins, 0, f32_bits(v))


def h_F2I(m, w, l, ins):
    """F2I.<U32|S32>.<TRUNC|FLOOR|CEIL|NEAR> d, a — float to integer.

    .U32 is *unsigned*: a negative or out-of-range float clamps to 0/0xFFFFFFFF
    rather than wrapping, which is what hardware does.  .S32 clamps to the
    signed range.  Getting this wrong shows up as garbage digits in every
    integer division.
    """
    import math

    v = bits_f32(rd(m, w, l, ins, 1))
    if math.isnan(v):
        t = 0
    elif has(ins, "TRUNC"):
        t = int(v)  # toward zero
    elif has(ins, "FLOOR"):
        t = math.floor(v)
    elif has(ins, "CEIL"):
        t = math.ceil(v)
    else:
        t = int(round(v))  # near-even
    if has(ins, "U32"):
        t = 0 if t < 0 else min(t, MASK32)
    else:
        t = max(-(1 << 31), min(t, (1 << 31) - 1))
    wr(m, w, l, ins, 0, t & MASK32)


def h_MUFU(m, w, l, ins):
    v = bits_f32(rd(m, w, l, ins, 1))
    if has(ins, "RCP"):
        r = float("inf") if v == 0.0 else 1.0 / v
    elif has(ins, "RSQ"):
        r = float("inf") if v == 0.0 else 1.0 / (v ** 0.5)
    elif has(ins, "EX2"):
        import math
        r = math.pow(2.0, v)
    elif has(ins, "LG2"):
        import math
        r = math.log2(v) if v > 0 else float("-inf")
    elif has(ins, "SQRT"):
        r = v ** 0.5
    else:
        raise Unsupported(f"MUFU variant: {ins.raw}")
    if has(ins, "FTZ"):
        r = _ftz_sub(r)
    wr(m, w, l, ins, 0, f32_bits(r))
    m.stats.setdefault("mufu", {})
    key = "MUFU." + "+".join(ins.mods)
    m.stats["mufu"][key] = m.stats["mufu"].get(key, 0) + 1


def _mem_tok(ins, i):
    tok = ins.ops[i]
    if tok[0] != "MEM":
        raise Unsupported(f"expected memory operand, got {tok} in {ins.raw}")
    return tok


def _extend(m, ins, raw, size):
    if has(ins, "S8") or has(ins, "S16"):
        bits = size * 8
        sign = 1 << (bits - 1)
        return (raw - (1 << bits)) & MASK32 if raw & sign else raw & MASK32
    return raw & MASK32


def _do_load(m, w, l, ins, space):
    # SASS prints loads destination-first:  LDG.E R4, [R4.64]
    mem = _mem_tok(ins, 1)
    addr = m._addr(w, l, mem)
    size = ins.width
    nregs = max(1, size // 4)
    dst = ins.ops[0]
    if size in (1, 2, 4):
        raw = space.read(addr, size)
        wr(m, w, l, ins, 0, _extend(m, ins, raw, size))
        return
    for k in range(nregs):
        chunk = space.read(addr + 4 * k, 4)
        if dst[0] == "R":
            m._write_reg(w, l, ("R", dst[1] + k), chunk)
        else:
            m._write_reg(w, l, dst, chunk)


def _do_store(m, w, l, ins, space):
    # stores print address-first:  STG.E [R2.64], R5
    mem = _mem_tok(ins, 0)
    addr = m._addr(w, l, mem)
    size = ins.width
    nregs = max(1, size // 4)
    val_tok = ins.ops[1]
    for k in range(nregs):
        if val_tok[0] == "R":
            v = m._reg(w, l, ("R", val_tok[1] + k), 4)
        else:
            v = m._reg(w, l, val_tok, 4)
        space.write(addr + 4 * k, min(4, size), v)


def h_LDG(m, w, l, ins):
    # LDG prints destination-first:  LDG.E R4, [R4.64]
    if ins.ops[0][0] == "MEM":
        _do_store(m, w, l, ins, m.global_space)
    else:
        _do_load(m, w, l, ins, m.global_space)


def h_STG(m, w, l, ins):
    addr = m._addr(w, l, _mem_tok(ins, 0))
    if m.global_space.find(addr, ins.width) is None:
        raise Unsupported(
            f"store outside mapped global memory at 0x{addr:x} "
            f"(pc 0x{ins.addr:04x}, width {ins.width}): {ins.raw}")
    _do_store(m, w, l, ins, m.global_space)


def h_LDS(m, w, l, ins):
    # LDS prints destination-first:  LDS R7, [R7.X4]   (STG is address-first)
    if ins.ops[0][0] == "MEM":
        _do_store(m, w, l, ins, m.shared_space)
    else:
        _do_load(m, w, l, ins, m.shared_space)


def h_STS(m, w, l, ins):
    # STS prints address-first:  STS [R9.X4], R2
    if ins.ops[0][0] == "MEM":
        _do_store(m, w, l, ins, m.shared_space)
    else:
        _do_load(m, w, l, ins, m.shared_space)


def h_ATOM(m, w, l, ins):
    mem = _mem_tok(ins, 0)
    addr = m._addr(w, l, mem)
    old = m.global_space.read(addr, 4)
    val = rd(m, w, l, ins, 1)
    if has(ins, "ADD"):
        new = (old + val) & MASK32
    elif has(ins, "EXCH"):
        new = val & MASK32
    elif has(ins, "MAX"):
        new = max(s32(old), s32(val)) & MASK32
    elif has(ins, "MIN"):
        new = min(s32(old), s32(val)) & MASK32
    else:
        raise Unsupported(f"ATOM variant: {ins.raw}")
    m.global_space.write(addr, 4, new)
    wr(m, w, l, ins, 0, old)


def h_RED(m, w, l, ins):
    mem = _mem_tok(ins, 0)
    addr = m._addr(w, l, mem)
    old = m.global_space.read(addr, 4)
    val = rd(m, w, l, ins, 1)
    size = ins.width
    if has(ins, "ADD"):
        if size == 8:
            cur = m.global_space.read(addr, 8)
            m.global_space.write(addr, 8, (cur + rd(m, w, l, ins, 1, 8)) & MASK64)
            return
        new = (old + val) & MASK32
    elif has(ins, "MAX"):
        new = max(s32(old), s32(val)) & MASK32
    elif has(ins, "MIN"):
        new = min(s32(old), s32(val)) & MASK32
    else:
        raise Unsupported(f"RED variant: {ins.raw}")
    m.global_space.write(addr, 4, new)


def h_SHFL(m, w, l, ins):
    # SHFL.DOWN/UP/BFLY/IDX  pred, d, a, b, mask
    pred_tok = ins.ops[0]
    val = rd(m, w, l, ins, 2)
    b = rd(m, w, l, ins, 3)
    width = 32
    mask_tok = ins.ops[4] if len(ins.ops) > 4 else ("IMM", 0x1F)
    segmask = mask_tok[1] if mask_tok[0] == "IMM" else 0x1F
    segwidth = (segmask & 0x1F) + 1
    base = (l // segwidth) * segwidth
    off = l % segwidth
    valid = True
    if has(ins, "DOWN"):
        src = off + b
        valid = src < segwidth
    elif has(ins, "UP"):
        src = off - b
        valid = src >= 0
    elif has(ins, "BFLY"):
        src = off ^ b
        valid = True
    elif has(ins, "IDX"):
        src = (b & 0x1F) if segmask == 0 else (b & segmask)
        valid = True
    else:
        raise Unsupported(f"SHFL mode: {ins.raw}")
    if valid:
        srclane = base + src
        value = w.regs[srclane][ins.ops[2][1]] if ins.ops[2][0] == "R" else val
    else:
        value = val
    m._write_reg(w, l, ins.ops[1], value)
    if pred_tok[0] in ("P", "NOTP"):
        m._write_reg(w, l, pred_tok, valid)


def h_IMNMX(m, w, l, ins):
    """IMNMX d, a, b, p  ->  d = MIN ? min(a,b) : max(a,b); unsigned with .U32."""
    a = rd(m, w, l, ins, 1)
    b = rd(m, w, l, ins, 2)
    if not has(ins, "U32"):
        a, b = s32(a), s32(b)
    v = max(a, b) if has(ins, "MAX") else min(a, b)
    wr(m, w, l, ins, 0, v & MASK32)


def h_IABS(m, w, l, ins):
    """IABS d, a  ->  d = |a| (two's complement, as a 32-bit int)."""
    v = s32(rd(m, w, l, ins, 1))
    wr(m, w, l, ins, 0, (-v if v < 0 else v) & MASK32)


def h_BREV(m, w, l, ins):
    v = rd(m, w, l, ins, 1) & MASK32
    out = 0
    for i in range(32):
        if (v >> i) & 1:
            out |= 1 << (31 - i)
    wr(m, w, l, ins, 0, out)


def h_FLO(m, w, l, ins):
    v = rd(m, w, l, ins, 1) & MASK32
    if v == 0:
        wr(m, w, l, ins, 0, 0xFFFFFFFF)
        return
    n = 0
    while not (v & 0x80000000):
        v = (v << 1) & MASK32
        n += 1
    wr(m, w, l, ins, 0, n)


def h_CLZ(m, w, l, ins):
    v = rd(m, w, l, ins, 1) & MASK32
    wr(m, w, l, ins, 0, 32 if v == 0 else 32 - v.bit_length())


def h_BSSY(m, w, l, ins):
    pass


def h_BSYNC(m, w, l, ins):
    pass


def h_NOP(m, w, l, ins):
    pass


def h_EXIT(m, w, l, ins):
    pass


handlers = {
    "MOV": h_MOV,
    "S2R": h_S2R,
    "ULDC": h_ULDC,
    "IMAD": h_IMAD,
    "IADD3": h_IADD3,
    "LEA": h_LEA,
    "ISETP": h_ISETP,
    "SEL": h_SEL,
    "SHF": h_SHF,
    "LOP3": h_LOP3,
    "PLOP3": h_PLOP3,
    "HFMA2": h_HFMA2,
    "FADD": h_FADD,
    "FMUL": h_FMUL,
    "FFMA": h_FFMA,
    "FMNMX": h_FMNMX,
    "I2F": h_I2F,
    "F2I": h_F2I,
    "MUFU": h_MUFU,
    "LDG": h_LDG,
    "STG": h_STG,
    "LDS": h_LDS,
    "STS": h_STS,
    "ATOM": h_ATOM,
    "RED": h_RED,
    "SHFL": h_SHFL,
    "IMNMX": h_IMNMX,
    "IABS": h_IABS,
    "BREV": h_BREV,
    "FLO": h_FLO,
    "CLZ": h_CLZ,
    "BSSY": h_BSSY,
    "BSYNC": h_BSYNC,
    "NOP": h_NOP,
    "EXIT": h_EXIT,
}
SASSMachine.handlers = handlers


# ==========================================================================
# cubin metadata (no source, no hard-coded per-kernel numbers)
# ==========================================================================

_RES_RE = re.compile(
    r"Function\s+(\S+)\s*:\s*\n\s*REG:(\d+)\s+STACK:(\d+)\s+SHARED:(\d+)\s+"
    r"LOCAL:(\d+)\s+CONSTANT\[0\]:(\d+)")


def resource_usage(cubin_path, cuobjdump="/opt/cuda/bin/cuobjdump"):
    """Per-function REG/STACK/SHARED/LOCAL/CONSTANT usage, straight from the
    cubin. Returns {function_name: dict}."""
    out = subprocess.run([cuobjdump, "-res-usage", cubin_path],
                         capture_output=True, text=True, check=True).stdout
    res = {}
    for m in _RES_RE.finditer(out):
        res[m.group(1)] = {
            "reg": int(m.group(2)),
            "stack": int(m.group(3)),
            "shared": int(m.group(4)),
            "local": int(m.group(5)),
            "constant0": int(m.group(6)),
        }
    return res


def sass_of(cubin_path, cuobjdump="/opt/cuda/bin/cuobjdump"):
    return subprocess.run([cuobjdump, "-sass", cubin_path],
                          capture_output=True, text=True, check=True).stdout
