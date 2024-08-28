#include "CudaCheckError.h"
#include "CudaGraph.h"
#include "CudaSSSP.h"

constexpr int INF = INT_MAX >> 1;

__global__ static void init(nodeId_t nodeNum, nodeId_t sourceId, weight_t* cost)
{
    int tId = blockDim.x * blockIdx.x + threadIdx.x;
    if (tId < nodeNum) {
        if (tId == sourceId) {
            cost[tId] = 0;
        } else {
            cost[tId] = INF;
        }
    }
}

__global__ static void bellmanFord(nodeId_t nodeNum, nodeId_t* edgeIndicesStart, nodeId_t* edgeIndicesEnd,
    weight_t* ea, weight_t* weights, weight_t* cost)
{
    int tId = blockDim.x * blockIdx.x + threadIdx.x;

    if (tId < nodeNum) {
        nodeId_t startIndex = edgeIndicesStart[tId];
        nodeId_t endIndex = edgeIndicesEnd[tId];
        for (nodeId_t i = startIndex; i < endIndex; i++) {
            nodeId_t neighbor = ea[i];
            weight_t weight = weights[i];
            // 如果当前节点可达
            if (cost[tId] != INF) {
                // 防止读者写者问题
                atomicMin(&cost[neighbor], cost[tId] + weight);
            }
        }
    }
}

std::vector<weight_t> cudaSSSP(LinkGraph& graph, nodeId_t sourceId)
{
    CudaLinkGraph cudaLG(graph);
    nodeId_t nodeNum = graph.getNodeNum();
    nodeId_t edgeNum = graph.getEdgeNum();

    int block = 1024;
    int grid = (nodeNum + 1023) / 1024;

    weight_t* d_cost;
    checkError(cudaMalloc(&d_cost, nodeNum * sizeof(weight_t)));

    init<<<grid, block>>>(nodeNum, sourceId, d_cost);
    checkError(cudaDeviceSynchronize());

    // 不考虑负权边
    for (nodeId_t i = 0; i < nodeNum; i++) {
        bellmanFord<<<grid, block>>>(nodeNum, cudaLG.d_edgeIndicesStart, cudaLG.d_edgeIndicesEnd, cudaLG.d_ea,
            cudaLG.d_weights, d_cost);
        checkError(cudaDeviceSynchronize());
    }

    std::vector<weight_t> cost(nodeNum);
    checkError(cudaMemcpy(cost.data(), d_cost, nodeNum * sizeof(weight_t), cudaMemcpyDeviceToHost));
    checkError(cudaFree(d_cost));

    return cost;
}
