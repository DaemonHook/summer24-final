#include "CudaCheckError.h"
#include "CudaPageRank.h"
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

constexpr static float eps = 1e-6;

/// @brief 将矩阵初始化为pagerank初始状态
__global__ static void initMatrix(float* mat, int n)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            sum += matGet(mat, n, i, id);
        }

        for (int i = 0; i < n; i++) {
            if (fabsf(sum) > eps) {
                matGet(mat, n, i, id) /= sum;
            } else {
                matGet(mat, n, i, id) = 1.0f / n;
            }
        }
    }
}

__global__ static void initRank(float* rank, int n)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < n) {
        rank[id] = 1.0f / n;
    }
}

__global__ static void saveRank(float* r, float* rLast, int n)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < n) {
        rLast[id] = r[id];
    }
}

__global__ static void mulLast(float* mat, float* rLast, float* r, int n)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < n) {
        float sum = 0.0f;
        for (int j = 0; j < n; j++) {
            sum += rLast[j] * matGet(mat, n, id, j);
        }

        r[id] = sum;
    }
}

std::vector<float> cudaPageRank(MatrixGraph& graph)
{
    nodeId_t n = graph.getNodeNum();

    std::vector<float> h_mat(n * n);
    std::vector<float> h_r(n);
    // 连通为1，不连通为0
    std::transform(graph._mat.begin(), graph._mat.end(), h_mat.begin(), [](weight_t i) { if (i > 0) return 1; else return 0; });

    int block = 1024;
    int grid = (n + 1023) / block;

    float* d_mat;
    checkError(cudaMalloc(&d_mat, n * n * sizeof(float)));
    checkError(cudaMemcpy(d_mat, h_mat.data(), n * n * sizeof(float), cudaMemcpyHostToDevice));
    initMatrix<<<grid, block>>>(d_mat, n);

    // debug
    // checkError(cudaDeviceSynchronize());
    // checkError(cudaMemcpy(h_mat.data(), d_mat, n * n * sizeof(float), cudaMemcpyDeviceToHost));
    // std::cout << "MAT:\n";
    // std::for_each(h_mat.begin(), h_mat.end(), [](float f) { std::cout << f << ' '; });
    // std::cout << std::endl;

    float* d_r;
    checkError(cudaMalloc(&d_r, n * sizeof(float)));
    initRank<<<grid, block>>>(d_r, n);

    // debug
    // checkError(cudaDeviceSynchronize());
    // checkError(cudaMemcpy(h_r.data(), d_r, n * sizeof(float), cudaMemcpyDeviceToHost));
    // std::cout << "rank:\n";
    // std::for_each(h_r.begin(), h_r.end(), [](float f) { std::cout << f << ' '; });
    // std::cout << std::endl;

    float* d_rLast;
    checkError(cudaMalloc(&d_rLast, n * sizeof(float)));

    checkError(cudaDeviceSynchronize());

    int maxiter = 1000;
    while (maxiter--) {
        saveRank<<<grid, block>>>(d_r, d_rLast, n);
        checkError(cudaDeviceSynchronize());

        mulLast<<<grid, block>>>(d_mat, d_rLast, d_r, n);
        checkError(cudaDeviceSynchronize());

        thrust::device_ptr<float> d_rLastP(d_rLast);
        float result = thrust::reduce(d_rLastP, d_rLastP + n);

        if (result < eps) {
            checkError(cudaMemcpy(h_r.data(), d_r, n * sizeof(float), cudaMemcpyDeviceToHost));
            return h_r;
        }
    }
    checkError(cudaMemcpy(h_r.data(), d_r, n * sizeof(float), cudaMemcpyDeviceToHost));
    return h_r;
}