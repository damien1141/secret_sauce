"""Lightweight concolic SASS tracer for the MoE kernel.

Parses cuobjdump -sass output, builds a basic-block CFG, and simulates
register flow for a CONCRETE input.  Tracks which input-derived values flow
into each STG store index, and reports the store (index, value) pairs so we
can see whether the index is a recoverable function of the inputs.
"""
import re, subprocess, sys

def parse_sass(path):
    """Return list of (addr, mnemonic, operands, raw) per instruction."""
    instrs = []
    cur = None
    for line in open(path):
        m = re.match(r'\s*/\*([0-9a-f]+)\*/\s+(.+?)\s+/\*.*\*/\s*$', line)
        if m:
            addr = int(m.group(1), 16)
            body = m.group(2).strip()
            # split mnemonic from operands
            parts = body.split(None, 1)
            mnem = parts[0]
            ops = parts[1] if len(parts) > 1 else ''
            instrs.append((addr, mnem, ops, body))
    return instrs

def main():
    sass_path = sys.argv[1]
    instrs = parse_sass(sass_path)
    print(f"parsed {len(instrs)} instructions")
    # Show the store instructions and their addresses
    stores = [(a, m, o) for a, m, o, _ in instrs if m.startswith("STG")]
    print(f"\n{len(stores)} STG stores:")
    for a, m, o in stores:
        print(f"  0x{a:04x}  {m}  {o}")

if __name__ == "__main__":
    main()
