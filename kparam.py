#!/usr/bin/env python3
"""Parse EIATTR_KPARAM_INFO from a CUDA 13.4 cubin's .nv.info section.

The KPARAM entry (attribute 0x17, SVAL, 12 bytes) is:
  [0:2]   index
  [2:4]   ordinal
  [4:6]   offset (bytes from arg start)
  [6:8]   padding
  [8:12]  cbank_info: bit 0x10000 -> is_cbank (scalar in constant bank),
                       bit 0x20000 -> space (0 = pointer, 1 = scalar)

This is the authoritative launch-argument layout for a closed-source cubin.
Works on CUDA 11.x and 13.4 ELF ABI.
"""
import struct, subprocess


def kparam_layout(cubin):
    """Return list of (ordinal, offset, is_pointer) sorted by offset."""
    data = open(cubin, "rb").read()
    e_shoff = struct.unpack_from("<Q", data, 0x28)[0]
    e_shentsize = struct.unpack_from("<H", data, 0x3a)[0]
    e_shnum = struct.unpack_from("<H", data, 0x3c)[0]
    e_shstrndx = struct.unpack_from("<H", data, 0x3e)[0]

    def sh(i):
        return data[e_shoff + i * e_shentsize: e_shoff + (i + 1) * e_shentsize]

    strtab_off = struct.unpack_from("<Q", sh(e_shstrndx), 0x18)[0]
    strtab_size = struct.unpack_from("<Q", sh(e_shstrndx), 0x20)[0]
    strtab = data[strtab_off:strtab_off + strtab_size]

    out = []
    for i in range(e_shnum):
        h = sh(i)
        name_off = struct.unpack_from("<I", h, 0)[0]
        name = strtab[name_off:strtab.index(b"\0", name_off)].decode()
        if not name.startswith(".nv.info."):
            continue
        off = struct.unpack_from("<Q", h, 0x18)[0]
        size = struct.unpack_from("<Q", h, 0x20)[0]
        blob = data[off:off + size]
        pos = 0
        while pos + 2 <= len(blob):
            fmt, attr = blob[pos], blob[pos + 1]
            if fmt in (1, 2, 3):
                pos += 4
            elif fmt == 4:
                vs = struct.unpack_from("<H", blob, pos + 2)[0]
                payload = blob[pos + 4:pos + 4 + vs]
                if attr == 0x17 and vs == 12:  # KPARAM
                    ordinal = struct.unpack_from("<H", payload, 2)[0]
                    offset = struct.unpack_from("<H", payload, 4)[0]
                    cbank = struct.unpack_from("<I", payload, 8)[0]
                    # 0x0021f000 = pointer (space bit 0x20000 set);
                    # 0x0011f000 = scalar (constant-bank arg).
                    # 0x21f000 = pointer (bit 0x200000 set);
                    # 0x11f000 = scalar (bit 0x100000 set).
                    is_ptr = bool(cbank & 0x200000)
                    out.append((ordinal, offset, is_ptr, cbank))
                pos += 4 + vs
            else:
                break
    out.sort(key=lambda t: t[1])
    return out


if __name__ == "__main__":
    import sys
    for f in sys.argv[1:]:
        print(f"=== {f} ===")
        for ordinal, offset, is_ptr, cbank in kparam_layout(f):
            print(f"  ordinal={ordinal} offset={offset:3d} "
                  f"ptr={is_ptr} cbank=0x{cbank:08x}")
