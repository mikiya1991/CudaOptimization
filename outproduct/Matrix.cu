#include <memory>
#include <Matrix.hpp>

Matrix::Matrix(int w, int h): width(w), height(h), stride(w)
{
    this->elements = static_cast<float *>(malloc(w * h * sizeof(float)));
    cudaMalloc(&this->gdata, this->stride * h * sizeof(float));
}


Matrix::~Matrix()
{
    cudaFree(&this->gdata);
    if (this->elements) {
        free(this->elements);
        this->elements = NULL;
    }
}

void Matrix::__todevice(void) {
    cudaMemcpy(this->gdata, this->elements, 
        this->stride * this->height * sizeof(float), cudaMemcpyHostToDevice);
}


void Matrix::__tohost(void) {
    cudaMemcpy(this->elements, this->gdata, 
                this->stride * this->height * sizeof(float),
                cudaMemcpyDeviceToHost);
} 


Matrix Matrix::eye(int size)
{

    Matrix A(size, size);

    auto pf = A.elements;

    for (int i = 0; i < A.height; i++) {
        for (int j = 0; j < A.width; j++) {
            if (i == j)
                pf[i * A.stride + j] = 1.0 * i;
        }
    }

    return A;
}
