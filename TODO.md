# Working SASS simulator — verified against the GPU

Goal: make the SASS simulator actually *execute* SASS and reproduce a kernel's
memory result byte-for-byte, checked against real hardware (RTX 4080, sm_89).

- [x] Inventory the SASS instruction set each target needs
- [x] `sass_emu.py` — parser, 32-lane warp, byte-addressable memory, barriers
- [x] CFG execution with real branch/predicate semantics and back-edges
- [x] `sass_verify.py` — run the same cubin on GPU and emulator, diff every byte
- [x] `sass_opcheck.py` — differential per-idiom test (GPU vs emulator)
- [x] Byte-match `vec_add` (elementwise), 3 sizes
- [x] Byte-match `custom` (shared mem, BAR.SYNC, FFMA chain), 3 sizes
- [x] Byte-match `moe_permute_prepare` (loops, binary search, scatter), 6 shapes
- [x] Byte-match `transpose` (2-D grid/block, padded shared memory)
- [x] Document results + the one instruction shape left unmodelled (README)
- [x] Update `README.zh-CN.md` (tables + translated simulator section)
