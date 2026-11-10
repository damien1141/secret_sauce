// Per-row softmax via warp shuffle (shfl_xor_sync) — from CUDA_Kernel_Samples/reduce/softmax_matrix/softmax_row_kernel2
// Input MxN, output MxN. Each block (one per row) computes row-max, sum of exp(x-max), then divides.
// Non-linear op (expf, fmaxf, division) — a real test of behavioral fidelity beyond FMA.

#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

#define CEIL(a, b) (((a) + (b) - 1) / (b))

__global__ void softmax_row_kernel2(float* input, float* output, int M, int N) {
    int laneId = threadIdx.x % warpSize;
    int row = blockIdx.x;
    if (row >= M) return;

    int iteration = CEIL(N, warpSize);

    float max_val = -FLT_MAX;
    for (int i = 0; i < iteration; i++) {
        int col = i * warpSize + laneId;
        max_val = (col < N) ? fmaxf(max_val, input[row * N + col]) : max_val;
    }
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        max_val = fmaxf(max_val, __shfl_xor_sync(0xFFFFFFFF, max_val, offset));
    }

    float sum = 0.0f;
    for (int i = 0; i < iteration; i++) {
        int col = i * warpSize + laneId;
        sum += (col < N) ? expf(input[row * N + col] - max_val) : 0.0f;
    }
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);
    }

    for (int i = 0; i < iteration; i++) {
        int col = i * warpSize + laneId;
        if (col < N) output[row * N + col] = expf(input[row * N + col] - max_val) / sum;
    }
}

extern "C" void launch_softmax_row(float* input, float* output, int M, int N) {
    dim3 grid(M);
    dim3 block(32);
    softmax_row_kernel2<<<grid, block>>>(input, output, M, N);
}