#pragma once

#include "CudaGraph.h"
#include "Graph.h"

/// @brief cuda并行的Prim算法，见 https://github.com/Dibyadarshan/GPU-Based-Fast-Minimum-Spanning-Tree
long long cudaPrim(std::vector<std::vector<std::pair<int, int>>>& adjList, int nodes, int edges);