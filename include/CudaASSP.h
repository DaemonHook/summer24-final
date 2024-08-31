#pragma once

#include "CudaGraph.h"
#include <vector>

std::vector<std::vector<weight_t>> CudaASSP(LinkGraph& graph);
std::vector<std::vector<weight_t>> cudaFloydWarshall(LinkGraph& graph);
