#include "CudaASSP.h"
#include "CudaCheckError.h"
#include "CudaSSSP.h"
#include <cuda_runtime.h>

std::vector<std::vector<weight_t>> CudaASSP(LinkGraph& graph)
{
    CudaLinkGraph cudaLG(graph);
    nodeId_t nodeNum = graph.getNodeNum();

    int grid = (nodeNum + 1023) / 1024;
    int block = 1024;

    weight_t* d_cost;
    checkError(cudaMalloc(&d_cost, nodeNum * sizeof(weight_t)));
    std::vector<std::vector<weight_t>> costs;

    // 依次求得每个点的SSSP
    for (nodeId_t i = 0; i < nodeNum; i++) {
        initCost<<<grid, block>>>(nodeNum, i, d_cost);
        checkError(cudaDeviceSynchronize());

        for (nodeId_t j = 0; j < nodeNum - 1; j++) {
            bellmanFord<<<grid, block>>>(nodeNum, cudaLG.d_edgeIndicesStart, cudaLG.d_edgeIndicesEnd, cudaLG.d_ea,
                cudaLG.d_weights, d_cost);
            checkError(cudaDeviceSynchronize());
        }

        std::vector<weight_t> curCost(nodeNum);
        checkError(cudaMemcpy(curCost.data(), d_cost, nodeNum * sizeof(weight_t), cudaMemcpyDeviceToHost));
        costs.push_back(std::move(curCost));
    }
    checkError(cudaFree(d_cost));
    return costs;
}

constexpr int INF = INT_MAX >> 1;
// CUDA 核函数，执行Floyd-Warshall算法的核心步骤
__global__ void floydWarshall(nodeId_t nodeNum, weight_t* dist, nodeId_t k)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i < nodeNum && j < nodeNum) {
        weight_t ikj = dist[i * nodeNum + k] + dist[k * nodeNum + j];
        if (ikj < dist[i * nodeNum + j]) {
            dist[i * nodeNum + j] = ikj;
        }
    }
}


__global__ void initDis(nodeId_t nodeNum, weight_t* cost)
{
    int tId = blockDim.x * blockIdx.x + threadIdx.x;
    if (tId < nodeNum) {
        cost[tId] = INF;
    }
}

__global__ void readGraph(nodeId_t nodeNum, nodeId_t* edgeIndicesStart, nodeId_t* edgeIndicesEnd,
    weight_t* ea, weight_t* weights, weight_t* cost)
{
    int tId = blockDim.x * blockIdx.x + threadIdx.x;

    if (tId < nodeNum) {
        nodeId_t startIndex = edgeIndicesStart[tId];
        nodeId_t endIndex = edgeIndicesEnd[tId];
        for (nodeId_t i = startIndex; i < endIndex; i++) {
            nodeId_t neighbor = ea[i];
            weight_t weight = weights[i];
            cost[tId*nodeNum+neighbor]=weight;
        }
    }
}
std::vector<std::vector<weight_t>> cudaFloydWarshall(LinkGraph& graph)
{
    CudaLinkGraph cudaLG(graph);

    nodeId_t nodeNum = graph.getNodeNum();


    int blockSize = 32; // 假设block大小为32x32
    dim3 block(blockSize, blockSize);
    dim3 grid((nodeNum + blockSize - 1) / blockSize, (nodeNum + blockSize - 1) / blockSize);


    weight_t* d_dist;
    
    checkError(cudaMalloc(&d_dist, nodeNum * nodeNum * sizeof(weight_t)));
    std::vector<weight_t> hostDist(nodeNum * nodeNum, INF);
    
    checkError(cudaMemcpy(d_dist, hostDist.data(), nodeNum * nodeNum * sizeof(weight_t), cudaMemcpyHostToDevice));

    int Grid = (nodeNum + 1023) / 1024;
    int Block = 1024;

    readGraph<<<Grid, Block>>>(nodeNum, cudaLG.d_edgeIndicesStart, cudaLG.d_edgeIndicesEnd, cudaLG.d_ea,
            cudaLG.d_weights, d_dist);

    // 迭代地执行Floyd-Warshall算法的核心步骤
    for (nodeId_t k = 0; k < nodeNum; k++) {
        floydWarshall<<<grid, block>>>(nodeNum, d_dist, k);
        checkError(cudaDeviceSynchronize());
    }

    checkError(cudaMemcpy(hostDist.data(), d_dist, nodeNum * nodeNum * sizeof(weight_t), cudaMemcpyDeviceToHost));
    checkError(cudaFree(d_dist));

    // 将一维数组转换为二维矩阵
    std::vector<std::vector<weight_t>> dist(nodeNum, std::vector<weight_t>(nodeNum));
    for (nodeId_t i = 0; i < nodeNum; i++) {
        for (nodeId_t j = 0; j < nodeNum; j++) {
            dist[i][j] = hostDist[i * nodeNum + j];
        }
    }

    return dist;
}

