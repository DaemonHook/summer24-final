#pragma once

#include "Graph.h"
#include <cuda_runtime.h>

/// @brief Cuda的bfs
/// @return 到其他点的距离（不考虑权重） 
std::vector<int> cudaBFS(LinkGraph& graph, nodeId_t sourceId);
