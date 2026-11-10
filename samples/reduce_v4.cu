// Warp-shuffle reduction to a single scalar — from CUDA_Kernel_Samples/reduce/sum/sum.cu (device_reduce_v4)
// Each block reduces its slice via shfl_down_sync, lane 0 atomicAdds into d_y.
// Output is a 1-element scalar (the total sum). Exercises the atomic-reduction path.

#include <cuda_runtime.h>

#define CEIL(a, b) (((a) + (b) - 1) / (b))

__global__ void device_reduce_v4(float* d_x, float* d_y, const int N) {
    __shared__ float s_y[32];
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int warpId = threadIdx.x / warpSize;
    int laneId = threadIdx.x % warpSize;

    float val = (idx < N) ? d_x[idx] : 0.0f;
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    if (laneId == 0) s_y[warpId] = val;
    __syncthreads();

    if (warpId == 0) {
        int warpNum = blockDim.x / warpSize;
        val = (laneId < warpNum) ? s_y[laneId] : 0.0f;
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
        if (laneId == 0) atomicAdd(d_y, val);
    }
}

extern "C" void launch_reduce_v4(float* d_x, float* d_y, int N) {
    const int BLOCK_SIZE = 128;
    dim3 block(BLOCK_SIZE);
    dim3 grid(CEIL(N, BLOCK_SIZE));
    device_reduce_v4<<<grid, block>>>(d_x, d_y, N);
}