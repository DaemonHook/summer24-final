#ifndef CUDACHECKERROR
#define CUDACHECKERROR

#include <cuda_runtime.h>

void checkError(cudaError_t err);

#endif