#include <Matrix.hpp>
#include <getopt.h>

int main(int argc, char *argv[])
{
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

    Matrix A(matsz, matsz);
    Matrix B(matsz, matsz);
    Matrix C(matsz, matsz);

    Matrix::matmul(A, B, C);

    if (!show_result)
        return 0;

    for (int i = 0; i < C.height; i++) {
        for (int j = 0; j < C.width; j++) 
            printf("%.0f ", C.elements[i * C.stride + j]);
        printf("\n");
    }
    printf("\n");
}