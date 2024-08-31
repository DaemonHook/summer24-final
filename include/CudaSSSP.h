#pragma once

#include "Common.h"
#include "Graph.h"
#include <vector>

__global__ void initCost(nodeId_t nodeNum, nodeId_t sourceId, weight_t* cost);

__global__ void bellmanFord(nodeId_t nodeNum, nodeId_t* edgeIndicesStart, nodeId_t* edgeIndicesEnd,
    weight_t* ea, weight_t* weights, weight_t* cost);

/// @brief Cuda的单源最短路径
/// @return 源点到每个点的最短路径长度
std::vector<weight_t> cudaBellmanFord(LinkGraph& graph, nodeId_t sourceId);