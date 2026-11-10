// Matrix-vector product, one row per block, warp-shuffle reduction — from CUDA_Kernel_Samples/gemv/sgemv_k32.cu
// A is MxK, x is K, y is M. Each block computes one output element y[row] = dot(A[row,*], x).
// Output is per-block (one value per row), like the custom kernel but 2D input.

#include <cuda_runtime.h>

#define CEIL(a, b) (((a) + (b) - 1) / (b))

__global__ void sgemv_k32(float* A, float* x, float* y, int M, int K) {
    int laneId = threadIdx.x % warpSize;
    int row = blockIdx.x;
    if (row >= M) return;

    float res = 0.0f;
    int kIteration = CEIL(K, warpSize);
    for (int i = 0; i < kIteration; i++) {
        int col = i * warpSize + laneId;
        res += (col < K) ? A[row * K + col] * x[col] : 0.0f;
    }
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        res += __shfl_down_sync(0xFFFFFFFF, res, offset);
    }
    if (laneId == 0) y[row] = res;
}

extern "C" void launch_sgemv_k32(float* A, float* x, float* y, int M, int K) {
    dim3 grid(M);
    dim3 block(32);
    sgemv_k32<<<grid, block>>>(A, x, y, M, K);
}