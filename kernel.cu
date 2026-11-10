// Original CUDA kernel — the "closed-source" target we are porting.
// Compiled to a standalone shared library, called from Python via ctypes.

extern "C" __global__ void custom_kernel(const float* a, const float* b, float* out, int n) {
    __shared__ float tile_a[256];
    __shared__ float tile_b[256];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;

    if (tid < n) tile_a[threadIdx.x] = a[tid]; else tile_a[threadIdx.x] = 0.0f;
    if (tid < n) tile_b[threadIdx.x] = b[tid]; else tile_b[threadIdx.x] = 0.0f;
    __syncthreads();

    float acc = 0.0f;
    for (int i = 0; i < 256; ++i) {
        acc += tile_a[i] * tile_b[i];
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    if (lane == 0 && tid < n) {
        out[blockIdx.x] = acc;
    }
}

extern "C" void launch_custom_kernel(const float* a, const float* b, float* out, int n) {
    const int kTile = 256;
    int blocks = (n + kTile - 1) / kTile;
    custom_kernel<<<blocks, kTile>>>(a, b, out, n);
}