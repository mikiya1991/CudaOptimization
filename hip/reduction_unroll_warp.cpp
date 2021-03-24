#include "hip/hip_runtime.h"
#define BLOCK_SIZE 128
#define DATALEN_PER_BLOCK (BLOCK_SIZE * 2)
#include <stdio.h>

/**
    this works, because
    last warp is like simd ? when is simd or not?
    volatile because each instruct deps on last result, why others not need ?
    **/
__device__ void warpReduce(volatile float *sdata, int tid)
{
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__global__ void reductionKernel(float *a, float *r)
{
    int blocksz = blockDim.x;
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    int i1 = DATALEN_PER_BLOCK * bid + tid;
    int i2 = i1 + blocksz;

    __shared__ float shared_data[BLOCK_SIZE];

    shared_data[tid] = a[i1] + a[i2];
    //shared_data[tid + blocksz] = a[i2];

    __syncthreads();

    for (int i = blocksz / 2; i > 32; i >>= 1) {
        if (tid < i) {
            shared_data[tid] += shared_data[tid + i];
        }

        __syncthreads();
    }

    if (tid < 32)
        warpReduce(shared_data, tid);

    r[bid] = shared_data[0];
}


float reduction(float *a, size_t len)
{
    int data_len_per_block = DATALEN_PER_BLOCK;
    float *dr = nullptr;
    float *da = nullptr;
    float *r = new float[data_len_per_block * sizeof(float)];
    size_t tlen = len;

    hipMalloc(&da, sizeof(float) * len);

    for (int k = 0; k < 20; k++) {
        len = tlen;
        hipMemcpy(da, a, sizeof(float) * len, hipMemcpyHostToDevice);

        while (len > data_len_per_block) {
            len /= data_len_per_block;

            dim3 threads(BLOCK_SIZE);
            dim3 grids(len);

            hipLaunchKernelGGL(reductionKernel, dim3(grids), dim3(threads), 0, 0, da, da);
        }
    }

    hipMemcpy(r, da, len * sizeof(float), hipMemcpyDeviceToHost);

    if (len > 0) {
        for (int i = 1; i < len; i++)
            r[0] += r[i];
    }

    int rr = r[0];

    hipFree(&da);
    hipFree(&dr);
    free(r);
    return rr;
}


int main(int argc, char *argv[])
{
    int len = 8192;

    if (argc > 1)
        len = atoi(argv[1]);

    printf("len %d\n", len);

    float *a = new float[len];

    for (int i = 0; i < len; i++)
        a[i] = 1.0;

    float r = reduction(a, len);
    printf("%.2f\n", r);
    return 0;
}
