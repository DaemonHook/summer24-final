#pragma once

#include "Common.h"
#include "Graph.h"
#include <cuda_runtime.h>

/// @brief 在CUDA显存中的LinkGraph
struct CudaLinkGraph {
    /// @brief 节点数量
    nodeId_t nodeNum;
    /// @brief 边数量
    long edgeNum;

    // 以下指针指向显存

    // 下面两个数组相当于va数组
    /// @brief 节点后继在ea中的开始索引
    nodeId_t* d_edgeIndicesStart;
    /// @brief 节点后继在ea中的结束索引
    nodeId_t* d_edgeIndicesEnd;

    /// @brief ea数组
    nodeId_t* d_ea;
    /// @brief 权重数组
    weight_t* d_weights;

    CudaLinkGraph(LinkGraph& graph);
    ~CudaLinkGraph();
};