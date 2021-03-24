#include "hip/hip_runtime.h"
#include <stdio.h>
#include <math.h>

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
__global__ void reductionKernel(float *a, float *r, int n)
{
    int tid = threadIdx.x;
    int index = blockIdx.x * blockSize * 2 + tid;
    int gridSize = gridDim.x * blockSize * 2;
    float rr = 0.0;

    __shared__ float shared_data[blockSize];

/*
 *    As `cascade algorithm` said:
 *    - a block is better with 256 threads;
 *    - a thread better do log(N) sums; sometimes 1024/2048 elements vs 256;
 *    - task should be processing wth N/log(N) threads;
 *
 *    G80 experience
 *    - more work per thread will hiding latency
 *    - More threads per block reduce kernel invoking time
 *    - few blocks will cause High kernel launch
 *    Best 64-256 blocks of 128 threads, 1024 - 4096 elements per thread
 */

    while (index < n) {
        rr += a[index] + a[index + blockSize];
        index += gridSize;
    }

    shared_data[tid] = rr;
    //shared_data[tid + blocksz] = a[i2];

    __syncthreads();

    if (blockSize >= 512) {{if (tid < 256) shared_data[tid] += shared_data[tid + 256];} __syncthreads();}
    if (blockSize >= 256) {{if (tid < 128) shared_data[tid] += shared_data[tid + 128];} __syncthreads();}
    if (blockSize >= 128) {{if (tid < 64) shared_data[tid] += shared_data[tid + 64];} __syncthreads();}

    if (tid < 32)
        warpReduce<blockSize>(shared_data, tid);
    if (tid == 0)
        r[blockIdx.x] = shared_data[0];
}


float reduction(float *a, size_t len, int blocksz, int gridsz)
{
    float *da;
    float *r = new float[blocksz * sizeof(float)];
    size_t tlen = len;
    int tgridsz = gridsz;
    size_t data_per_block = blocksz * 2;

    hipMalloc(&da, sizeof(float) * len);

    for (int k = 0; k < 20; k++) {
        len = tlen;
        gridsz = tgridsz;
        hipMemcpy(da, a, sizeof(float) * len, hipMemcpyHostToDevice);

        //do {
            dim3 threads(blocksz);
            dim3 grids(gridsz);

/*
 *            int maxGridSz = len / (data_per_block);
 *
 *            if (gridsz > maxGridSz)
 *                gridsz = maxGridSz;
 */


            printf("gridsz %d\n", grids.x);

            switch (blocksz) {
                case 512:
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(reductionKernel<512>), dim3(grids), dim3(threads), 0, 0, da, da, len);
                    break;
                case 256:
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(reductionKernel<256>), dim3(grids), dim3(threads), 0, 0, da, da, len);
                    break;
                case 128:
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(reductionKernel<128>), dim3(grids), dim3(threads), 0, 0, da, da, len);
                    break;
                case 64:
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(reductionKernel<64>), dim3(grids), dim3(threads), 0, 0, da, da, len);
                    break;
                case 32:
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(reductionKernel<32>), dim3(grids), dim3(threads), 0, 0, da, da, len);
                    break;
                default:
                    printf("invalid block size, skip\n");
                    break;
            }

            //len = gridsz;
        //} while (len > data_per_block);
    }

    //hipMemcpy(r, da, len * sizeof(float), hipMemcpyDeviceToHost);

    /*
     *if (len > 0) {
     *    for (int i = 1; i < len; i++)
     *        r[0] += r[i];
     *}
     */

    int rr = r[0];

    hipFree(&da);
    free(r);
    return rr;
}


int main(int argc, char *argv[])
{
    int len = 4194304;
    int blocksz = 128;
    int gridsz;

    if (argc < 4) {
        printf("usage: %s [len] [blocksz] [gridsz]\n", argv[0]);
        return -1;
    }

    len = atoi(argv[1]);
    blocksz = atoi(argv[2]);
    gridsz = atoi(argv[3]);

    printf("len %d blocksz %d gridsz %d elemets_per_thread %d\n", len, blocksz, gridsz, len/(blocksz * gridsz));

    float *a = new float[len];

    for (int i = 0; i < len; i++)
        a[i] = 1.0;

    float r = reduction(a, len, blocksz, gridsz);
    printf("%.2f\n", r);
    return 0;
}
