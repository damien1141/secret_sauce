// Reference implementation of cuBLAS's transpose_readWrite_alignment_kernel
// (cublas_fatbin.2.sm_80.cubin). Written from the SASS structure:
// shared-memory tiled transpose with bank-conflict avoidance via padding.
//
// This is the GROUND TRUTH we verify the generated Triton against.
// The generated kernel never sees this file.

extern "C" __global__ void transpose_kernel(
    const float* in, float* out, int rows, int cols)
{
    const int TILE = 16;
    __shared__ float shared[TILE + 1][TILE + 1];  // padding avoids bank conflicts

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int in_col = bx * TILE + tx;
    int in_row = by * TILE + ty;

    if (in_row < rows && in_col < cols) {
        shared[ty][tx] = in[in_row * cols + in_col];
    }
    __syncthreads();

    int out_col = by * TILE + tx;
    int out_row = bx * TILE + ty;
    if (out_row < cols && out_col < rows) {
        out[out_row * rows + out_col] = shared[tx][ty];
    }
}

extern "C" void launch_transpose_kernel(
    const float* in, float* out, int rows, int cols)
{
    const int TILE = 16;
    dim3 blocks((cols + TILE - 1) / TILE, (rows + TILE - 1) / TILE, 1);
    dim3 threads(TILE, TILE, 1);
    transpose_kernel<<<blocks, threads>>>(in, out, rows, cols);
}