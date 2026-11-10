// Real production kernel from sgl-project/sglang (Apache 2.0).
// Extracted from kernels/jit/csrc/moe/moe_permute_prepare.cu — the kernel
// body is self-contained CUDA; the TVM-FFI host glue is replaced with a
// plain extern "C" launcher so port.py can drive it.
#include <cuda_runtime.h>
#include <algorithm>

namespace sglang {
__device__ __forceinline__ int32_t lower_bound(const int32_t* __restrict__ data, int32_t n, int32_t target) {
  int32_t lo = 0, hi = n;
  while (lo < hi) {
    int32_t mid = lo + (hi - lo) / 2;
    if (data[mid] < target) lo = mid + 1; else hi = mid;
  }
  return lo;
}

__global__ void moe_permute_prepare_kernel(
    const int32_t* __restrict__ sorted_topk_ids,
    const int64_t* __restrict__ reorder_ids,
    void* __restrict__ expert_offsets,
    int32_t* __restrict__ src2dst,
    int32_t num_experts, int32_t numel,
    bool use_int64_offset, bool is_ep) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  int32_t neg_count = lower_bound(sorted_topk_ids, numel, 0);
  for (int e = tid; e <= num_experts; e += stride) {
    int32_t offset;
    if (e < num_experts) offset = lower_bound(sorted_topk_ids, numel, e) - neg_count;
    else offset = numel - neg_count;
    if (use_int64_offset) reinterpret_cast<int64_t*>(expert_offsets)[e] = offset;
    else reinterpret_cast<int32_t*>(expert_offsets)[e] = offset;
  }
  for (int i = tid; i < numel; i += stride) src2dst[reorder_ids[i]] = i - neg_count;
}
}

extern "C" void launch_moe_permute_prepare(
    const int32_t* sorted_topk_ids, const int64_t* reorder_ids,
    void* expert_offsets, int32_t* src2dst,
    int32_t num_experts, int32_t numel, bool use_int64_offset, bool is_ep) {
  constexpr int threads = 256;
  int num_blocks = std::max(1, (std::max(numel, num_experts + 1) + threads - 1) / threads);
  sglang::moe_permute_prepare_kernel<<<num_blocks, threads>>>(
      sorted_topk_ids, reorder_ids, expert_offsets, src2dst,
      num_experts, numel, use_int64_offset, is_ep);
}
