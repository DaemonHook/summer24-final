#include "CudaASSP.h"
#include <cuda_runtime.h>
#include "CudaCheckError.h"

/// @brief 内层的并行函数
/// @param cost 当前的距离矩阵
/// @param mat 邻接矩阵
/// @return 
__host__ void inner(nodeId_t nodeNum, weight_t* cost, weight_t* mat)
{
}

std::vector<std::vector<weight_t>> CudaASSP(MatrixGraph& graph)
{
    CudaMatGraph cudaMG(graph);
    nodeId_t nodeNum = graph.getNodeNum();

    weight_t* d_cost;
    checkError(cudaMalloc(&d_cost, 


    return std::vector<std::vector<weight_t>>();
}