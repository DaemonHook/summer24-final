#pragma once

#include <cuda_runtime.h>

#define checkError(err)                                                 \
    if (err != cudaSuccess) {                                           \
        fprintf(stderr, "%s", __FILE__);                                \
        fprintf(stderr, " line %d", __LINE__);                          \
        fprintf(stderr, ": CUDA Runtime Error: %s\n", cudaGetErrorString(err)); \
        exit(-1);                                                       \
    }
