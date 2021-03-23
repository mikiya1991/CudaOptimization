#include <memory>

class Matrix{
public:
    int width;
    int height;
    int stride; 
    float *elements;
    float* gdata;
    Matrix(int w, int h);
    virtual ~Matrix();

    void __todevice(void);
    void __tohost(void);

    static Matrix eye(int size);
    static int matmul(Matrix &A, Matrix &B, Matrix &C);
};