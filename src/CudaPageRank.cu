#include "CudaCheckError.h"
#include "CudaPageRank.h"
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <utility>

using namespace std;

constexpr static float eps = 1e-6;

__global__ void manageMatrix(float* mat, int n)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        float sum = 0.0;
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

__global__ void initRank(float* r, int n)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < n) {
        r[id] = (1 / (float)n);
    }
}

__global__ void storeRank(float* r, float* rLast, int n)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        rLast[id] = r[id];
    }
}

__global__ void matmul(float* mat, float* r, float* rLast, int n)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < n) {
        float sum = 0.0;
        for (int j = 0; j < n; ++j) {
            sum += rLast[j] * mat[id * n + j];
        }
        r[id] = sum;
    }
}

__global__ void rankDiff(float* r, float* rLast, int n)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < n) {
        rLast[id] = abs(rLast[id] - r[id]);
    }
}

__global__ void initPairArray(std::pair<float, int>* rNodes, float* r, int n)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < n) {
        rNodes[id].first = r[id];
        rNodes[id].second = id + 1;
    }
}

void powerMethod(float* graph, float* r, int n, int max_iter = 1000)
{

    float* r_last = (float*)malloc(n * sizeof(float));

    float* g_graph;
    cudaMalloc(&g_graph, sizeof(float) * n * n);
    cudaMemcpy(g_graph, graph, sizeof(float) * n * n, cudaMemcpyHostToDevice);

    float* d_r;
    cudaMalloc(&d_r, sizeof(float) * n);

    float* d_rLast;
    cudaMalloc(&d_rLast, sizeof(float) * n);

    int block = 32;
    int grid = (n + 1023) / block;

    initRank<<<grid, block>>>(d_r, n);
    cudaDeviceSynchronize();

    while (max_iter--) {

        storeRank<<<grid, block>>>(d_r, d_rLast, n);
        cudaDeviceSynchronize();

        matmul<<<grid, block>>>(g_graph, d_r, d_rLast, n);
        cudaDeviceSynchronize();

        rankDiff<<<grid, block>>>(d_r, d_rLast, n);
        cudaDeviceSynchronize();

        
        cudaMemcpy(r_last, d_rLast, n * sizeof(float), cudaMemcpyDeviceToHost);
        float result = thrust::reduce(r_last, r_last + n);

        if (result < eps) {
            cudaMemcpy(r, d_r, n * sizeof(float), cudaMemcpyDeviceToHost);
            return;
        }
    }
    cudaMemcpy(r, d_r, n * sizeof(float), cudaMemcpyDeviceToHost);
    return;
}

std::vector<float> cudaPageRank(MatrixGraph& graph)
{
    nodeId_t n = graph.getNodeNum();
    vector<float> h_mat(n * n);

    // 连通为1，不连通为0
    std::transform(graph._mat.begin(), graph._mat.end(), h_mat.begin(),
        [](weight_t i) { if (i > 0) return 1.0f; else return 0.0f; });
    
    vector<float> r(n);
    powerMethod(h_mat.data(), r.data(), n);
    return r;
}