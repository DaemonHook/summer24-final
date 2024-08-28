#include "Common.h"
#include "Graph.h"
#include <vector>

/// @brief Cuda的单源最短路径
/// @return 源点到每个点的最短路径长度
std::vector<weight_t> cudaSSSP(LinkGraph& graph, nodeId_t sourceId);