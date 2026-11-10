// Reference kernel — simple vector add, no shared memory, no warp shuffle.
// Used to verify the pipeline handles the non-redundant case correctly.

extern "C" __global__ void vec_add(const float* a, const float* b, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

extern "C" void launch_vec_add(const float* a, const float* b, float* out, int n) {
    const int kTile = 256;
    int blocks = (n + kTile - 1) / kTile;
    vec_add<<<blocks, kTile>>>(a, b, out, n);
}