import sass_emu, struct, sys
cubin='build/opcheck.cubin'
func=sys.argv[1]; a=int(sys.argv[2]); b=int(sys.argv[3])
sass = sass_emu.sass_of(cubin)
m = sass_emu.SASSMachine(sass, func=func)
m.set_launch((1,1,1),(1,1,1))
OUT=0x20000000
m.load_bytes(OUT, b'\x00'*48)
m.set_params([('i32',a&0xFFFFFFFF),('i32',b&0xFFFFFFFF),('ptr',OUT)])
log=[]
orig=sass_emu.SASSMachine._exec_group
def ex(self,w,lanes,ins):
    log.append((hex(ins.addr), ins.raw))
    return orig(self,w,lanes,ins)
sass_emu.SASSMachine._exec_group=ex
m.run()
for x in log: print(x)
print("OUT:", struct.unpack('<12i', m.read_bytes(OUT,48)))
