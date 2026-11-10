// Naive single-precision matmul — from CUDA_Kernel_Samples/sgemm/src/kernel1.cu
// No shared memory, no tiling: one FMA chain over the full K dimension.
// Output is 2D elementwise (C[id_y*N + id_x] = alpha*dot + beta*C).
// Wrapped with an extern "C" launcher so port.py / bench.py can drive it.

#include <cuda_runtime.h>
#include <stdio.h>

#define CEIL(a, b) (((a) + (b) - 1) / (b))

__global__ void sgemm_v1(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    int id_x = blockIdx.x * blockDim.x + threadIdx.x;
    int id_y = blockIdx.y * blockDim.y + threadIdx.y;

    float tmp = 0.;
    for (int i = 0; i < K; i++) {
        tmp += A[id_y * K + i] * B[i * N + id_x];
    }
    if (id_x < N && id_y < M) {
        C[id_y * N + id_x] = alpha * tmp + beta * C[id_y * N + id_x];
    }
}

extern "C" void launch_sgemm_v1(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C) {
    dim3 block(16, 16);
    dim3 grid(CEIL(M, 16), CEIL(N, 16));
    sgemm_v1<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}