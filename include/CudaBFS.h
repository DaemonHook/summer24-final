#pragma once

#include "Graph.h"
#include <cuda_runtime.h>

std::vector<int> cudaBFS(LinkGraph& graph, nodeId_t sourceId);
