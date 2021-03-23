#include "hip/hip_runtime.h"
#include <cstdlib>
#include <cstring>
#include <cstdio>

__global__ void vecadd(float *A, float *B, float *C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;


    if (i < N) {
        //printf("%d %.2f %.2f\n", i, A[i], B[i]);
        C[i] = A[i] + B[i];
    }

    //printf("blockDim %d %d %d i %d \n", blockDim.x, blockIdx.x, threadIdx.x, i);
}


int main(int argc, char *argv[])
{
    float *h_a, *h_b, *h_c;
    size_t n = 1024;

    if (argc > 1)
        n = atoi(argv[1]);

    float *da, *db, *dc;
    size_t size = n * sizeof(float);


    h_a = static_cast<float *>(malloc(size));
    h_b = static_cast<float *>(malloc(size));
    h_c = static_cast<float *>(malloc(size));

    for (int i = 0; i < n; i++) {
        h_a[i] = i * 2.0f;
        h_b[i] = i * 2.0f + 1.0;

        printf("%.2f %.2f, ", h_a[i], h_b[i]);
    }
    printf("\n");

    hipMalloc(&da, size);
    hipMalloc(&db, size);
    hipMalloc(&dc, size);

    hipMemcpy(da, h_a, size, hipMemcpyHostToDevice);
    hipMemcpy(db, h_b, size, hipMemcpyHostToDevice);
    //hipMemcpy(dc, h_c, size, hipMemcpyHostToDevice);

    int threadPerBlock = 256;
    int blockPerGrid = (n + threadPerBlock - 1) / threadPerBlock;

    hipLaunchKernelGGL(vecadd, dim3(blockPerGrid), dim3(threadPerBlock), 0, 0, da, db, dc, n);
    hipMemcpy(h_c, dc, size, hipMemcpyDeviceToHost);

    for (int i = 0; i < n; i++)
        printf("%.2f ", h_c[i]);
    printf("\n");

    hipFree(&da);
    hipFree(&db);
    hipFree(&dc);

    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}