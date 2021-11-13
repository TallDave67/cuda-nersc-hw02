
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

int main_matrix_mul_shared();
int main_matrix_mul_global();
int main_stencil_1d();

int main()
{
    //printf("********** main_matrix_mul_shared\n");
    //int ret_shared = main_matrix_mul_shared();
    //printf("********** main_matrix_mul_global\n");
    //int ret_global = main_matrix_mul_global();
    printf("********** main_stencil_1d\n");
    int ret_stencil = main_stencil_1d();

    int ret = 0;
    //ret |= ret_shared;
    //ret |= ret_global;
    ret |= ret_stencil;

    return ret;
}
