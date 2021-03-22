#include <Matrix.hpp>

#define BLOCK_X_SIZE 64
#define BLOCK_Y_SIZE 16

__device__ float get_ele(float *m, int stride, int y, int x)
{
    return *(m + stride * y + x);
}


__device__ void set_ele(float *m, int stride, int y, int x, float v)
{
    *(m + stride * y + x) = v;
}


__global__ void MatMulKernel(const Matrix A, const Matrix B, Matrix C)
{
    int id = threadIdx.x;
    int gridy = blockIdx.y;
    int gridx = blockIdx.x;

    int row = id / BLOCK_Y_SIZE;
    int col = id % BLOCK_Y_SIZE;

    int stepcount = A.width / BLOCK_Y_SIZE;
    float c[BLOCK_Y_SIZE];

    for (int i = 0; i < BLOCK_Y_SIZE; i++)
        c[i] = 0.0f;

    for (int k = 0; k < stepcount; k++)  {
        __shared__ float As[BLOCK_Y_SIZE][BLOCK_Y_SIZE];

        float *tileA = A.gdata + A.stride * (BLOCK_Y_SIZE * gridy + row) + BLOCK_Y_SIZE * k;
        float *Bs = B.gdata + B.stride * BLOCK_Y_SIZE * k + BLOCK_X_SIZE * gridx;

        //load A sub to shared_memory

        As[row][col] = *(tileA + col);
        As[row][col + 1] = *(tileA + col + 1);
        As[row][col + 2] = *(tileA + col + 2);
        As[row][col + 3] = *(tileA + col + 3);

        __syncthreads();

        for (int i = 0; i < BLOCK_Y_SIZE; i++) {
            float Acol[BLOCK_Y_SIZE]; //get As [:, col_i].
            float b = get_ele(Bs, B.stride, i, id); //get Bs (row_i, id).

            for (int j = 0; j < BLOCK_Y_SIZE; j++)
                Acol[j] = As[j][i];

            for (int j = 0; j < BLOCK_Y_SIZE; j++)
                c[j] += b * Acol[j]; // result in Cs [:, id]
        }
        __syncthreads();
    }

    float *Cs =  C.gdata + gridy * BLOCK_Y_SIZE * C.stride + gridx * BLOCK_X_SIZE;

    for (int i = 0; i < BLOCK_Y_SIZE; i++)
        set_ele(Cs, C.stride, i, id, c[i]);
}


int Matrix::matmul(Matrix &A, Matrix &B, Matrix &C)
{
    dim3 dimBlock(BLOCK_X_SIZE);
    dim3 dimGrid(C.height / BLOCK_Y_SIZE, C.width / BLOCK_X_SIZE);

    A.__todevice();
    B.__todevice();

    for (int i = 0; i < 10; i++)
        MatMulKernel<<<dimGrid, dimBlock>>>(A, B, C);
    C.__tohost();
}
