#include <stdio.h>

/**
    this works, because 
    last warp is like simd ? when is simd or not?
    volatile because each instruct deps on last result, why others not need ?
    **/
template <int blockSize>
__device__ void warpReduce(volatile float *sdata, int tid)
{
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}


template <int blockSize>
__global__ void reductionKernel(float *a, float *r)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    int i1 = blockSize * 2 * bid + tid;
    int i2 = i1 + blockSize;

    __shared__ float shared_data[blockSize];

    shared_data[tid] = a[i1] + a[i2];
    //shared_data[tid + blocksz] = a[i2];

    __syncthreads();

    if (blockSize >= 512) {{if (tid < 256) shared_data[tid] += shared_data[tid + 256];} __syncthreads();}
    if (blockSize >= 256) {{if (tid < 128) shared_data[tid] += shared_data[tid + 128];} __syncthreads();}
    if (blockSize >= 128) {{if (tid < 64) shared_data[tid] += shared_data[tid + 64];} __syncthreads();}

    if (tid < 32)
        warpReduce<blockSize>(shared_data, tid);

    r[bid] = shared_data[0];
}


float reduction(float *a, size_t len, int blocksz)
{
    float *da;
    float *r = new float[blocksz * sizeof(float)];
    size_t tlen = len;
    size_t data_per_block = blocksz * 2;

    cudaMalloc(&da, sizeof(float) * len);

    for (int k = 0; k < 20; k++) {
        len = tlen;
        cudaMemcpy(da, a, sizeof(float) * len, cudaMemcpyHostToDevice);

        while (len > (data_per_block)) {
            len /= data_per_block;

            dim3 threads(blocksz);
            dim3 grids(len);

            switch (blocksz) {
                case 512:
                    reductionKernel<512><<<grids, threads>>>(da, da);
                    break;
                case 256:
                    reductionKernel<256><<<grids, threads>>>(da, da);
                    break;
                case 128:
                    reductionKernel<128><<<grids, threads>>>(da, da);
                    break;
                case 64:
                    reductionKernel<64><<<grids, threads>>>(da, da);
                    break;
                case 32:
                    reductionKernel<32><<<grids, threads>>>(da, da);
                    break;
                default:
                    printf("invalid block size, skip\n");
                    break;
            }
        }
    }

    cudaMemcpy(r, da, len * sizeof(float), cudaMemcpyDeviceToHost);

    if (len > 0) {
        for (int i = 1; i < len; i++)
            r[0] += r[i];
    }

    int rr = r[0];

    cudaFree(&da);
    free(r);
    return rr;
}


int main(int argc, char *argv[])
{
    int len = 4194304;
    int blocksz = 128;

    if (argc > 1)
        len = atoi(argv[1]);
    if (argc > 2)
        blocksz = atoi(argv[2]);

    printf("len %d blocksz %d\n", len, blocksz);

    float *a = new float[len];

    for (int i = 0; i < len; i++)
        a[i] = 1.0;

    float r = reduction(a, len, blocksz);
    printf("%.2f\n", r);
    return 0;
}