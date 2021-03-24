#include "hip/hip_runtime.h"
#include <stdio.h>

template<int blockSize>
__global__ void reductionKernel(float *a, float *r)
{
    int tid = threadIdx.x;
    int i1 = blockSize * blockIdx.x + tid;

    __shared__ float shared_data[blockSize];

    shared_data[tid] = a[i1];
    __syncthreads();

    for (int s = blockSize/2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
        r[blockIdx.x] = shared_data[0];
}


float reduction(float *a, size_t len, int blockSize)
{
    float *dr = nullptr;
    float *da = nullptr;
    float *r = new float[blockSize];
    size_t tlen = len;

    hipMalloc(&da, sizeof(float) * len);

    for (int k = 0; k < 20; k++) {
        len = tlen;
        hipMemcpy(da, a, sizeof(float) * len, hipMemcpyHostToDevice);

        while (len > blockSize) {
            len /= blockSize;

            printf("len %d\n", len);

            dim3 threads(blockSize);
            dim3 grids(len);

            switch (blockSize) {
            case 512:
                hipLaunchKernelGGL(HIP_KERNEL_NAME(reductionKernel<512>), dim3(grids), dim3(threads), 0, 0, da, da);
                break;
            case 256:
                hipLaunchKernelGGL(HIP_KERNEL_NAME(reductionKernel<256>), dim3(grids), dim3(threads), 0, 0, da, da);
                break;
            case 128:
                hipLaunchKernelGGL(HIP_KERNEL_NAME(reductionKernel<128>), dim3(grids), dim3(threads), 0, 0, da, da);
                break;
            case 64:
                hipLaunchKernelGGL(HIP_KERNEL_NAME(reductionKernel<64>), dim3(grids), dim3(threads), 0, 0, da, da);
                break;
            case 32:
                hipLaunchKernelGGL(HIP_KERNEL_NAME(reductionKernel<32>), dim3(grids), dim3(threads), 0, 0, da, da);
                break;
            default:
                break;
            }
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
    int bs = 128;

    if (argc > 1)
        len = atoi(argv[1]);
    if (argc > 2)
        bs = atoi(argv[2]);

    printf("len %d blocksz %d\n", len, bs);

    float *a = new float[len];

    for (int i = 0; i < len; i++)
        a[i] = 1.0;

    float r = reduction(a, len, bs);
    printf("%.2f\n", r);
    return 0;
}
