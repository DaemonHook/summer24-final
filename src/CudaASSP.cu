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