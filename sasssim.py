#!/usr/bin/env python3
"""Concolic SASS simulator for the MoE permute kernel."""
import re, sys, json

def parse_sass(path):
    """Parse SASS, merging predicate guards (@P0, @!P1) into the following
    instruction as a leading operand rather than creating a phantom entry."""
    instrs = []
    pending_guard = None
    for line in open(path):
        m = re.match(r'\s*/\*([0-9a-f]+)\*/\s+(.+?)\s+/\*.*\*/\s*$', line)
        if m:
            addr = int(m.group(1), 16)
            body = m.group(2).strip()
            parts = body.split(None, 1)
            mnem = parts[0].split('.')[0]
            ops = parts[1] if len(parts) > 1 else ''
            if mnem.startswith("@"):
                # Predicate guard on the same line as the instruction:
                # "@!P1 BRA 0x180" -> guard="@!P1", mnem="BRA".
                # The real mnemonic is the next token.
                guard = mnem
                rest = parts[1].split(None, 1)
                mnem = rest[0].split(".")[0]
                ops = rest[1] if len(rest) > 1 else ""
                ops = guard + ", " + ops if ops else guard
                instrs.append((addr, mnem, ops, body))
                continue
            if pending_guard is not None:
                # Guard from the PREVIOUS line (rare form).
                ops = pending_guard + ", " + ops if ops else pending_guard
                pending_guard = None
            instrs.append((addr, mnem, ops, body))
    return instrs

def main():
    instrs = parse_sass(sys.argv[1])
    print(f"parsed {len(instrs)} instructions")
    stores = [(a, m, o) for a, m, o, _ in instrs if m.startswith("STG")]
    print(f"{len(stores)} STG stores:")
    for a, m, o in stores:
        print(f"  0x{a:04x}  {m}  {o}")

if __name__ == "__main__":
    main()


# --- operand decoding -------------------------------------------------------
# SASS operand forms we need for the integer-only MoE kernel:
#   R<n>          register
#   R<n>.reuse    register with reuse flag
#   R<n>.64       64-bit register pair
#   c[0x0][0x184] constant memory bank
#   PT            predicate (true)
#   RZ            zero register
#   0x1           immediate
#   [R6.64]       address: register pair
#   @P0           predicate guard


def reg(operand):
    """Return register number from 'R12' / 'R12.reuse' / 'R12.64'.

    Returns -1 for non-register operands so the caller can detect misses
    without crashing the register dict.
    """
    m = re.match(r'R(\d+)', operand)
    return int(m.group(1)) if m else -1


def is_pred(operand):
    return operand in ("PT", "true", "false") or operand.startswith("P")


def parse_operands(ops_str):
    """Split operand string into a list of operand tokens."""
    ops = []
    for o in ops_str.split(","):
        o = o.strip().split(";")[0].strip()
        if o:
            ops.append(o)
    return ops


# --- symbolic value --------------------------------------------------------
# A value is either a concrete int or a symbolic expression tree.
# Expression nodes: ("input", bank, idx) for a loaded input element,
# ("const", n), ("reg", r), ("add", a, b), ("sub", a, b), ("mul", a, b),
# ("and", a, b), ("or", a, b), ("shl", a, b), ("sra", a, b), ("mov", a).


class Sym:
    __slots__ = ("kind", "args", "val")

    def __init__(self, kind, args=(), val=None):
        self.kind = kind
        self.args = args
        self.val = val

    def concrete(self):
        return self.kind == "const"

    def __repr__(self):
        if self.kind == "const":
            return str(self.val)
        if self.kind == "reg":
            return f"R{self.args}"
        if self.kind == "input":
            return f"in[{self.args[0]}][{self.args[1]}]"
        return f"{self.kind}({', '.join(map(repr, self.args))})"


def const(n):
    return Sym("const", val=n & 0xFFFFFFFFFFFFFFFF)


def regval(r):
    return Sym("reg", (r,))


def input_elem(bank, idx):
    return Sym("input", (bank, idx))


def binop(kind, a, b):
    return Sym(kind, (a, b))


def eval_sym(s, regs, cbank, inputs):
    """Evaluate a symbolic value to a concrete int."""
    if s.kind == "const":
        return s.val
    if s.kind == "tid":
        return s.val  # thread id (0 for single-thread simulation)
    if s.kind == "ctaid":
        return s.val  # block id (0)
    if s.kind == "reg":
        return eval_sym(regs[s.args[0]], regs, cbank, inputs)
    if s.kind == "input":
        bank, idx = s.args
        return inputs.get((bank, idx), 0)
    a = eval_sym(s.args[0], regs, cbank, inputs)
    b = eval_sym(s.args[1], regs, cbank, inputs)
    if s.kind == "add":
        return (a + b) & 0xFFFFFFFFFFFFFFFF
    if s.kind == "sub":
        return (a - b) & 0xFFFFFFFFFFFFFFFF
    if s.kind == "mul":
        return (a * b) & 0xFFFFFFFFFFFFFFFF
    if s.kind == "and":
        return (a & b) & 0xFFFFFFFFFFFFFFFF
    if s.kind == "or":
        return (a | b) & 0xFFFFFFFFFFFFFFFF
    if s.kind == "shl":
        return (a << (b & 0x3F)) & 0xFFFFFFFFFFFFFFFF
    if s.kind == "sra":
        return (a >> (b & 0x3F)) & 0xFFFFFFFFFFFFFFFF
    if s.kind == "mov":
        return a
    return 0


# --- instruction execution -------------------------------------------------
# The MoE kernel uses only integer ops.  We execute each instruction over
# symbolic values, threading the concrete input through LDG loads so the
# store indices are traced back to their input sources.


def imm(operand):
    """Parse an immediate operand ('0x1', '0x2', '-R0')."""
    operand = operand.split(";")[0].strip()
    if operand.startswith("0x") or operand.startswith("-0x"):
        return int(operand, 16)
    try:
        return int(operand)
    except ValueError:
        return None


def src_sym(op, regs, cbank, inputs):
    """Resolve an operand to a symbolic value."""
    if op is None:
        return const(0)
    if op == "RZ":
        return const(0)
    if op == "PT":
        return const(1)
    m = re.match(r'R(\d+)(?:\.reuse)?$', op)
    if m:
        return regs[int(m.group(1))]
    m = re.match(r'R(\d+)\.64$', op)
    if m:
        lo = regs[int(m.group(1))]
        hi = regs.get(int(m.group(1)) + 1, const(0))
        return binop("or", binop("shl", hi, const(32)), lo)
    m = re.match(r'c\[0x[0-9a-f]+\]\[0x([0-9a-f]+)\]$', op)
    if m:
        idx = int(m.group(1), 16)
        if idx in cbank:
            return const(cbank[idx])
        key = "0x" + m.group(1)
        if key in cbank:
            return const(cbank[key])
        return const(0)
    return const(0)


def execute(instr, regs, cbank, inputs, pred, stores):
    """Execute one instruction.  Returns new predicate state."""
    addr, mnem, ops_str, _ = instr
    ops = parse_operands(ops_str)
    guard = None
    if ops and ops[0].startswith("@"):
        guard = ops[0]
        ops = ops[1:]
    _addr = addr

    def run():
        nonlocal pred
        if mnem == "IMAD":
            # IMAD.IADD d, a, b, c  -> d = a + b + c
            # IMAD.WIDE d, a, b, c  -> d = a*b + c  (wide multiply-add)
            # IMAD.MOV.U32 d, RZ, RZ, c  -> d = c  (MOV form)
            d = ops[0]
            if "WIDE" in ops_str:
                sa = src_sym(ops[1], regs, cbank, inputs)
                sb = src_sym(ops[2], regs, cbank, inputs)
                sc = src_sym(ops[3], regs, cbank, inputs)
                regs[reg(d)] = binop("add", binop("mul", sa, sb), sc)
            elif len(ops) == 4:
                sa = src_sym(ops[1], regs, cbank, inputs)
                sb = src_sym(ops[2], regs, cbank, inputs)
                sc = src_sym(ops[3], regs, cbank, inputs)
                regs[reg(d)] = binop("add", binop("add", sa, sb), sc)
            elif len(ops) == 3:
                regs[reg(d)] = src_sym(ops[2], regs, cbank, inputs)
        elif mnem == "IADD3":
            d, a, b, c = ops[0], ops[1], ops[2], ops[3]
            sa = src_sym(a, regs, cbank, inputs)
            sb = src_sym(b, regs, cbank, inputs)
            sc = src_sym(c, regs, cbank, inputs)
            regs[reg(d)] = binop("add", binop("add", sa, sb), sc)
        elif mnem == "ISETP":
            # ISETP.GE.AND P, PT, a, b, PT  -> P = (a >= b)
            p, _, a, b, _ = ops[0], ops[1], ops[2], ops[3], ops[4]
            sa = src_sym(a, regs, cbank, inputs)
            sb = src_sym(b, regs, cbank, inputs)
            pred[p[1:]] = eval_sym(sa, regs, cbank, inputs) >= eval_sym(sb, regs, cbank, inputs)
        elif mnem == "LEA":
            # LEA d, P, base, c, scale  -> d = base + c*scale
            # LEA.HI.X d, P, base, c, scale, P2  (6 operands)
            d = ops[0]
            base = ops[2]
            c = ops[3]
            scale = ops[4] if len(ops) > 4 else "0x1"
            sb = src_sym(base, regs, cbank, inputs)
            sc = src_sym(c, regs, cbank, inputs)
            ss = imm(scale) or 1
            regs[reg(d)] = binop("add", sb, binop("mul", sc, const(ss)))
        elif mnem == "LDG":
            # LDG.E.64.CONSTANT R10, [R4.64]  -> load from address
            # The address register holds the full byte address (base+offset),
            # so look up the loaded value by absolute address in the input
            # memory map.  This is what lets store indices be traced back to
            # the input array elements that produced them.
            d, addr = ops[0], ops[1]
            m = re.match(r'\[(R\d+)\.64\]$', addr)
            if m:
                base_reg = reg(m.group(1))
                base = regs[base_reg]
                addr_val = eval_sym(base, regs, cbank, inputs)
                if addr_val in inputs:
                    regs[reg(d)] = const(inputs[addr_val])
                else:
                    regs[reg(d)] = base  # trace as the address root
        elif mnem == "STG":
            # STG [addr], value
            addr_op, val_op = ops[0], ops[1]
            m = re.match(r'\[(R\d+)\.64\]$', addr_op)
            if m:
                base = regs[reg(m.group(1))]
                val = regs[reg(val_op)]
                idx = eval_sym(base, regs, cbank, inputs)
                v = eval_sym(val, regs, cbank, inputs)
                stores.append((_addr, reg(m.group(1)), idx, v, base, val))
        elif mnem == "S2R":
            # S2R R3, SR_CTAID.X  -> R3 = block id
            # S2R R0, SR_TID.X   -> R0 = thread id
            d, sr = ops[0], ops[1]
            if "TID" in sr:
                regs[reg(d)] = Sym("tid", (), val=0)
            elif "CTAID" in sr:
                regs[reg(d)] = Sym("ctaid", (), val=0)
            elif "WARPID" in sr:
                regs[reg(d)] = Sym("warpid", (), val=0)
            else:
                regs[reg(d)] = const(0)
        elif mnem == "SEL":
            d, p, a, b = ops[0], ops[1], ops[2], ops[3]
            sa = src_sym(a, regs, cbank, inputs)
            sb = src_sym(b, regs, cbank, inputs)
            regs[reg(d)] = sa if pred.get(p[1:], False) else sb
        elif mnem == "SHF":
            d, a, b, _ = ops[0], ops[1], ops[2], ops[3]
            sa = src_sym(a, regs, cbank, inputs)
            sb = src_sym(b, regs, cbank, inputs)
            regs[reg(d)] = binop("shl", sa, sb)
        elif mnem == "LOP3":
            d, a, b, _ = ops[0], ops[1], ops[2], ops[3]
            sa = src_sym(a, regs, cbank, inputs)
            sb = src_sym(b, regs, cbank, inputs)
            regs[reg(d)] = binop("and", sa, sb)
        elif mnem == "ULDC":
            # ULDC.64 UR6, c[0x0][0x118]  -> load uniform const
            d, c_op = ops[0], ops[1]
            m = re.match(r'c\[0x[0-9a-f]+\]\[0x([0-9a-f]+)\]$', c_op)
            if m:
                regs[reg(d)] = cbank.get(int(m.group(1), 16), const(0))
        elif mnem in ("NOP", "EXIT", "BSSY", "BSYNC"):
            pass
        elif mnem == "BRA":
            pass
        else:
            # Unknown op: best-effort, mark destination as opaque.
            if len(ops) >= 1 and re.match(r'R\d+', ops[0]):
                regs[reg(ops[0])] = const(0)

    if guard is None or pred.get(guard[1:], False):
        run()
    return pred


# --- CFG construction and simulation ---------------------------------------
def build_cfg(instrs):
    """Return (blocks, addr_to_block).  blocks: {addr: [instr, ...]}."""
    blocks = {}
    cur = None
    for ins in instrs:
        addr, mnem, _, _ = ins
        if mnem in ("BRA", "EXIT", "@!P0 BRA", "@P0 BRA"):
            # terminator: close current block
            if cur is not None:
                blocks.setdefault(cur, []).append(ins)
            cur = None
        else:
            if cur is None:
                cur = addr
            blocks.setdefault(cur, []).append(ins)
    return blocks


def simulate(instrs, inputs, cbank):
    """Run the kernel for one thread with concrete inputs.

    inputs: {(bank, idx): value}  for LDG loads.
    cbank:   {idx: value}         for c[0x0][idx] constant memory.
    Returns list of store records.
    """
    regs = {i: const(0) for i in range(256)}
    pred = {f"P{i}": False for i in range(8)}
    stores = []
    # Linear sweep with back-edge handling: follow BRA targets up to a bound.
    # Build address -> index map
    by_addr = {a: i for i, (a, _, _, _) in enumerate(instrs)}
    i = 0
    steps = 0
    max_steps = 500000
    # Linear sweep that always executes every instruction it lands on and
    # follows BRA targets.  No visited-skip: that short-circuited the loop
    # body and prevented the pointer setup from propagating into the stores.
    while i < len(instrs) and steps < max_steps:
        steps += 1
        ins = instrs[i]
        addr, mnem, _, _ = ins
        execute(ins, regs, cbank, inputs, pred, stores)
        if mnem == "BRA":
            # Operand string is "@!P1 BRA 0x180" — extract the hex target.
            m = re.search(r'0x([0-9a-f]+)', ops_str)
            if m:
                target = int(m.group(1), 16)
                if target in by_addr:
                    i = by_addr[target]
                    continue
        i += 1
    return stores, regs, steps








# --- proper CFG-based execution -------------------------------------------
# A linear sweep re-executes instructions and corrupts registers across
# loop iterations.  This builds real basic blocks and executes each block
# once per loop iteration, carrying register state across back-edges.


def build_blocks(instrs):
    """Split into basic blocks.  Returns {start_addr: [instr, ...]}."""
    targets = set()
    for ins in instrs:
        addr, mnem, _, _ = ins
        if mnem == "BRA":
            m = re.search(r'0x([0-9a-f]+)', ins[3])
            if m:
                targets.add(int(m.group(1), 16))
    blocks = {}
    cur_addr = None
    cur = []
    for ins in instrs:
        addr, mnem, _, _ = ins
        if cur and (addr in targets or mnem in ("BRA", "EXIT", "BSYNC", "BSSY")):
            blocks[cur_addr] = cur
            cur_addr = None
            cur = []
        if cur_addr is None:
            cur_addr = addr
            cur = []
        cur.append(ins)
        if mnem in ("BRA", "EXIT"):
            blocks[cur_addr] = cur
            cur_addr = None
            cur = []
    if cur:
        blocks[cur_addr] = cur
    return blocks


def block_successors(block, blocks):
    """Return list of successor block addresses."""
    last = block[-1]
    mnem = last[1]
    succ = []
    if mnem == "BRA":
        m = re.search(r'0x([0-9a-f]+)', last[3])
        if m:
            t = int(m.group(1), 16)
            if t in blocks:
                succ.append(t)
    elif mnem not in ("EXIT", "BSYNC", "BSSY"):
        # Fall-through: the address of the instruction after this block's
        # last instruction.  BSYNC/BSSY are barriers, not branches — they
        # fall through too, but we treat them as block ends for clarity.
        last_addr = block[-1][0]
        addrs = sorted(blocks)
        i = addrs.index(block[0][0]) if block[0][0] in addrs else -1
        if i >= 0 and i + 1 < len(addrs):
            succ.append(addrs[i + 1])
    elif mnem in ("BSYNC", "BSSY"):
        # Barrier sync: fall through to the next block.
        addrs = sorted(blocks)
        i = addrs.index(block[0][0]) if block[0][0] in addrs else -1
        if i >= 0 and i + 1 < len(addrs):
            succ.append(addrs[i + 1])
    return succ


def execute_block(block, regs, cbank, inputs, pred, stores):
    """Execute every instruction in a block."""
    for ins in block:
        execute(ins, regs, cbank, inputs, pred, stores)
    return pred


def simulate_cfg(instrs, inputs, cbank, max_iter=2000000):
    """Execute the CFG with loop-natural semantics.

    Uses a worklist of (block_addr, iteration_count).  Back-edges increment
    the iteration count; the same block may execute many times, carrying
    register state across iterations.  This is what makes loop-carried
    values (binary search bounds, induction variables) propagate correctly.
    """
    blocks = build_blocks(instrs)
    regs = {i: const(0) for i in range(256)}
    pred = {f"P{i}": False for i in range(8)}
    stores = []
    # Path-sensitive worklist: each entry carries its OWN register/predicate
    # state, so the same block can execute many times with different values.
    # This is what makes loop-carried induction variables propagate.
    # Cap by total steps; the kernel's loops are bounded (numel=100,
    # num_experts=16) so the worklist terminates naturally.
    worklist = [(min(blocks), regs, pred, 0)]
    seen = set()
    steps = 0
    while worklist and steps < max_iter:
        steps += 1
        addr, regs, pred, it = worklist.pop(0)
        if addr not in blocks:
            continue
        # Deduplicate by (block, register state hash).  Without this the
        # worklist explodes: each conditional branch doubles the paths and
        # the loop body has many of them.
        state_key = (addr, _reg_hash(regs), _pred_hash(pred))
        if state_key in seen:
            continue
        seen.add(state_key)
        block = blocks[addr]
        execute_block(block, regs, cbank, inputs, pred, stores)
        for s in block_successors(block, blocks):
            next_it = it + 1 if s <= addr else it
            new_regs = dict(regs)
            new_pred = dict(pred)
            worklist.append((s, new_regs, new_pred, next_it))
    return stores, regs, steps


def _reg_hash(regs):
    """Stable hash of register state for deduplication.

    Uses repr of the concrete value of each non-trivial register, so the
    comparison is always between strings and never mixes types.
    """
    parts = []
    for k in sorted(regs):
        if k < 0:
            continue
        v = eval_sym(regs[k], regs, {}, {})
        if v != 0:
            parts.append(f"{k}:{v}")
    return tuple(parts)


def _pred_hash(pred):
    return tuple(sorted(f"{k}:{int(v)}" for k, v in pred.items()))


def trace_cfg(sass_path, inputs, cbank):
    instrs = parse_sass(sass_path)
    blocks = build_blocks(instrs)
    print(f"{len(blocks)} basic blocks")
    stores, regs, steps = simulate_cfg(instrs, inputs, cbank)
    print(f"simulated {steps} block executions, {len(stores)} stores")
    for a, base, idx, v, isym, vsym in stores:
        print(f"  0x{a:04x}  R{base}[{idx}] = {v}")
    return stores


def main():
    if len(sys.argv) < 2:
        print("usage: sasssim.py <sass> [input_json]")
        return 1
    inputs = {}
    cbank = {}
    if len(sys.argv) >= 3:
        with open(sys.argv[2]) as f:
            data = json.load(f)
        inputs = {int(k): v for k, v in data.get("inputs", {}).items()}
        cbank = {int(k, 16): v for k, v in data.get("cbank", {}).items()}
    trace_cfg(sys.argv[1], inputs, cbank)
    return 0


if __name__ == "__main__":
    sys.exit(main())
