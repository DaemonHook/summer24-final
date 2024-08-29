#include "CudaCheckError.h"
#include "CudaGraph.h"
#include <iostream>
#include <algorithm>

CudaLinkGraph::CudaLinkGraph(LinkGraph& memoryGraph)
{
    nodeNum = memoryGraph.getNodeNum();
    edgeNum = memoryGraph.getEdgeNum();

    // startEdgeIndices相当于原始va数组
    std::vector<nodeId_t> h_startEdgeIndices = memoryGraph.va;
    std::vector<nodeId_t> h_endEdgeIndices(nodeNum);

    for (nodeId_t i = 0; i < nodeNum; i++) {
        // 没有边的特殊处理
        if (h_startEdgeIndices[i] == NO_EDGE) {
            h_endEdgeIndices[i] = NO_EDGE;
            continue;
        }
        // 逐个去找终点
        nodeId_t curEdgeIndex = h_startEdgeIndices[i];
        nodeId_t nextNodeIndex = i + 1;
        while (nextNodeIndex < nodeNum && h_startEdgeIndices[nextNodeIndex] == NO_EDGE) {
            nextNodeIndex++;
        }
        // 如果超过了最后一个节点
        if (nextNodeIndex == nodeNum) {
            h_endEdgeIndices[i] = edgeNum;
        } else {
            h_endEdgeIndices[i] = h_startEdgeIndices[nextNodeIndex];
        }
    }
    
    checkError(cudaMalloc(&d_edgeIndicesStart, edgeNum * sizeof(nodeId_t)));
    checkError(cudaMemcpy(d_edgeIndicesStart, h_startEdgeIndices.data(), nodeNum * sizeof(nodeId_t), cudaMemcpyHostToDevice));
    checkError(cudaMalloc(&d_edgeIndicesEnd, edgeNum * sizeof(nodeId_t)));
    checkError(cudaMemcpy(d_edgeIndicesEnd, h_endEdgeIndices.data(), nodeNum * sizeof(nodeId_t), cudaMemcpyHostToDevice));
    checkError(cudaMalloc(&d_ea, edgeNum * sizeof(nodeId_t)));
    checkError(cudaMemcpy(d_ea, memoryGraph.ea.data(), edgeNum * sizeof(nodeId_t), cudaMemcpyHostToDevice));
    checkError(cudaMalloc(&d_weights, edgeNum * sizeof(weight_t)));
    checkError(cudaMemcpy(d_weights, memoryGraph.weights.data(), edgeNum * sizeof(nodeId_t), cudaMemcpyHostToDevice));
}

CudaLinkGraph::~CudaLinkGraph()
{
    checkError(cudaFree(d_edgeIndicesStart));
    checkError(cudaFree(d_edgeIndicesEnd));
    checkError(cudaFree(d_ea));
    checkError(cudaFree(d_weights));
}

CudaMatGraph::CudaMatGraph(MatrixGraph& graph)
{
    nodeNum = graph.getNodeNum();
    checkError(cudaMalloc(&d_mat, nodeNum * nodeNum * sizeof(weight_t)));
    checkError(cudaMemcpy(d_mat, graph._mat.data(), nodeNum * nodeNum * sizeof(weight_t), cudaMemcpyHostToDevice));
}

CudaMatGraph::~CudaMatGraph()
{
    checkError(cudaFree(d_mat));
}
