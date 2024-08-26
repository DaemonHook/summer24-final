#include "CudaCheckError.h"

#include <stdio.h>
#include <stdlib.h>

void checkError(cudaError_t err)
{
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}