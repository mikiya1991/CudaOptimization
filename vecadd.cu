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

    cudaMalloc(&da, size);
    cudaMalloc(&db, size);
    cudaMalloc(&dc, size);

    cudaMemcpy(da, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(db, h_b, size, cudaMemcpyHostToDevice);
    //cudaMemcpy(dc, h_c, size, cudaMemcpyHostToDevice);

    int threadPerBlock = 256;
    int blockPerGrid = (n + threadPerBlock - 1) / threadPerBlock;

    vecadd<<<blockPerGrid, threadPerBlock>>>(da, db, dc, n);
    cudaMemcpy(h_c, dc, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++)
        printf("%.2f ", h_c[i]);
    printf("\n");

    cudaFree(&da);
    cudaFree(&db);
    cudaFree(&dc);

    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}