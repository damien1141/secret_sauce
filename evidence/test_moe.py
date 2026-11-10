import torch, ctypes, subprocess, numpy as np
subprocess.run(["/opt/cuda/bin/nvcc","-shared","-Xcompiler=-fPIC","-O3","-arch=sm_80",
    "samples/real/moe_permute_prepare.cu","-o","build/moe_permute_prepare.so"],
    check=True, capture_output=True)
lib = ctypes.CDLL("build/moe_permute_prepare.so")
lib.launch_moe_permute_prepare.argtypes = [
    ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int64),
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
    ctypes.c_int32, ctypes.c_int32, ctypes.c_bool, ctypes.c_bool]

# Build a realistic MoE topk: sorted token->expert assignments
numel = 100
sorted_topk = np.array([0,0,0,1,1,2,2,2,2,3,3,4,4,4,5,5,5,6,6,7,7,7,7,8,8,9,9,9,9,9,
                        10,10,11,11,11,12,12,13,13,13,13,14,14,14,15,15,15,15,16,16,
                        17,17,17,18,18,18,18,19,19,19,20,20,20,21,21,21,21,22,22,22,
                        23,23,23,23,24,24,24,24,25,25,26,26,26,27,27,27,27,28,28,29,
                        29,29,29,29,30,30,31,31,31,31,32,32,33,33,33,34,34,35,35,35],
                       dtype=np.int32)
# reorder_ids: a permutation of 0..numel-1
reorder_ids = np.array([99,0,98,1,97,2,96,3,95,4,94,5,93,6,92,7,91,8,90,9,
                        89,10,88,11,87,12,86,13,85,14,84,15,83,16,82,17,81,18,80,19,
                        79,20,78,21,77,22,76,23,75,24,74,25,73,26,72,27,71,28,70,29,
                        69,30,68,31,67,32,66,33,65,34,64,35,63,36,62,37,61,38,60,39,
                        59,40,58,41,57,42,56,43,55,44,54,45,53,46,52,47,51,48,50,49],
                       dtype=np.int64)
num_experts = 36
expert_offsets = np.zeros(num_experts + 1, dtype=np.int32)
src2dst = np.zeros(numel, dtype=np.int32)

st = torch.from_numpy(sorted_topk).cuda()
ri = torch.from_numpy(reorder_ids).cuda()
eo = torch.from_numpy(expert_offsets).cuda()
s2d = torch.from_numpy(src2dst).cuda()

lib.launch_moe_permute_prepare(
    ctypes.cast(st.data_ptr(), ctypes.POINTER(ctypes.c_int32)),
    ctypes.cast(ri.data_ptr(), ctypes.POINTER(ctypes.c_int64)),
    ctypes.cast(eo.data_ptr(), ctypes.c_void_p),
    ctypes.cast(s2d.data_ptr(), ctypes.POINTER(ctypes.c_int32)),
    num_experts, numel, False, False)

# CPU reference
neg = int(np.searchsorted(sorted_topk, 0, side='left'))
ref_offsets = np.array([int(np.searchsorted(sorted_topk, e, side='left')) - neg
                        if e < num_experts else numel - neg for e in range(num_experts + 1)])
ref_src2dst = np.zeros(numel, dtype=np.int32)
for i in range(numel): ref_src2dst[reorder_ids[i]] = i - neg

eo_np = eo.cpu().numpy(); s2d_np = s2d.cpu().numpy()
print("expert_offsets match:", np.array_equal(eo_np, ref_offsets))
print("  cuda  :", eo_np[:6], "...")
print("  ref   :", ref_offsets[:6], "...")
print("src2dst match:", np.array_equal(s2d_np, ref_src2dst))
print("  cuda  :", s2d_np[:6], "...")
print("  ref   :", ref_src2dst[:6], "...")
