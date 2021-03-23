#include "hip/hip_runtime.h"
#include <cstdio>
#include <cstring>
#include <getopt.h>
//#include <pybind11.h>

#define BLOCK_SIZE 64
//#define CELL_SIZE 4
// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.stride + col)
typedef struct {
    int width;
    int height;
    int stride; 
    float* elements;
} Matrix;

// Get a matrix element
__device__ float GetElement(const Matrix A, int row, int col)
{
    return A.elements[row * A.stride + col];
}

// Set a matrix element
__device__ void SetElement(Matrix A, int row, int col,
                           float value)
{
    A.elements[row * A.stride + col] = value;
}

// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
 __device__ Matrix GetSubMatrix(Matrix A, int row, int col) 
{
    Matrix Asub;
    Asub.width    = BLOCK_SIZE;
    Asub.height   = BLOCK_SIZE;
    Asub.stride   = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row
                                         + BLOCK_SIZE * col];
    return Asub;
}


// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = d_A.stride = A.width;
    d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    hipMalloc(&d_A.elements, size);
    hipMemcpy(d_A.elements, A.elements, size,
               hipMemcpyHostToDevice);
    Matrix d_B;
    d_B.width = d_B.stride = B.width; d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    hipMalloc(&d_B.elements, size);
    hipMemcpy(d_B.elements, B.elements, size,
    hipMemcpyHostToDevice);

    // Allocate C in device memory
    Matrix d_C;
    d_C.width = d_C.stride = C.width;
    d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    hipMalloc(&d_C.elements, size);

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE/4, BLOCK_SIZE/4);
    dim3 dimGrid(B.width / BLOCK_SIZE, A.height / BLOCK_SIZE);
    for (int i = 0; i < 10; i++)
        hipLaunchKernelGGL(MatMulKernel, dim3(dimGrid), dim3(dimBlock), 0, 0, d_A, d_B, d_C);

    // Read C from device memory
    hipMemcpy(C.elements, d_C.elements, size,
               hipMemcpyDeviceToHost);

    // Free device memory
    hipFree(d_A.elements);
    hipFree(d_B.elements);
    hipFree(d_C.elements);
}

// Matrix multiplication kernel called by MatMul()
 __global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Each thread block computes one sub-matrix Csub of C
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);

    // Each thread computes one element of Csuba
    // by accumulating results into Cvalue

    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results

    float c[4][4];

    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            c[i][j] = 0.0f;

    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
        // Get sub-matrix Asub of A
        Matrix Asub = GetSubMatrix(A, blockRow, m);
        // Get sub-matrix Bsub of B
        Matrix Bsub = GetSubMatrix(B, m, blockCol);

        // Shared memory used to store Asub and Bsub respectively
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];



        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        for (int i = 0; i < (4); i++) 
            for (int j = 0; j < (4); j++) {
                As[row * 4 + i][col * 4 + j] = GetElement(Asub, row * 4 + i, col * 4 + j);
                Bs[row * 4 + i][col * 4 + j] = GetElement(Bsub, row * 4 + i, col * 4 + j);
            }

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();
        // Multiply Asub and Bsub together
        // for (int e = 0; e < BLOCK_SIZE; ++e)
        //     Cvalue += As[row][e] * Bs[e][col];

        float a[4][4];
        float b[4][4];

        for (int k = 0; k < (BLOCK_SIZE / 4); k++) {
            for (int i = 0; i < 4; i++) 
                for (int j = 0; j < 4; j++) {
                    a[i][j] = As[(row) * 4 + i][(k) * 4 + j];
                    b[i][j] = Bs[(k) * 4 + i][(col) * 4 + j];
                }

            for (int i = 0; i < 4; i++) //row
                for (int j = 0; j < 4; j++) //col
                    for (int l = 0; l < 4; l++)
                        c[i][j] += a[i][l] * b[l][j];
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write Csub to device memory
    // Each thread writes one element
    //SetElement(Csub, row, col, Cvalue);
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            SetElement(Csub, row*4 + i, col*4 + j, c[i][j]);
}


void __create_matrix(Matrix *m, int h, int w)
{
    m->width = w;
    m->height = h;
    m->stride = w;
    m->elements = static_cast<float *>(malloc(h * w * sizeof(float)));

    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            if (i==j)
                m->elements[i*w + j] = i;
        }
    }
}


int main(int argc, char *argv[])
{
    Matrix A, B, C;

    int matsz = 8192;
    int opt;
    int show_result = 0;

    while (-1 != (opt = getopt(argc, argv, "ps:"))) {
        switch (opt) {
        case 'p': 
            show_result = 1;
            break;
        case 's':
            matsz = atoi(optarg);
            break;
        default:
            printf("-p show result\n-s matrix size\n");
            break;
        }
    }

    __create_matrix(&A, matsz, matsz);
    __create_matrix(&B, matsz, matsz);
    __create_matrix(&C, matsz, matsz);

    MatMul(A, B, C);

    if (!show_result)
        return 0;
    for (int i = 0; i < C.height; i++) {
        for (int j = 0; j < C.width; j++)
            printf("%.0f ", C.elements[i * C.stride + j]);
        printf("\n");
    }

    return 0;
}