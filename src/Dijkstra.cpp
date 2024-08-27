#include "Graph.h"

/// @brief 邻接表-堆优化O(nlogn)
/// @param start 起始节点
/// @return
std::vector<nodeId_t> LinkGraph::dijkstra(nodeId_t start)
{
    nodeId_t inf = 0x3f3f3f3f;
    std::vector<nodeId_t> dist(this->vertexNum, inf);
    dist[start] = 0;
    std::priority_queue<std::pair<nodeId_t, nodeId_t>, std::vector<std::pair<nodeId_t, nodeId_t>>, std::greater<>> pq;
    pq.emplace(0, start);
    while (!pq.empty()) {
        auto [dis, node] = pq.top();
        pq.pop();
        if (dis > dist[node]) {
            continue;
        }
        // 获取 node 的邻居节点
        auto pointer = this->getSuccessors(node);
        while (pointer.end()) {
            nodeId_t neighbor_id = pointer.getId();
            // 待改动
            weight_t neighbor_weight = weights[0];
            nodeId_t new_dis = dis + neighbor_weight;
            if (new_dis < dist[neighbor_id]) {
                dist[neighbor_id] = new_dis;
                pq.emplace(new_dis, neighbor_id);
            }
            pointer.toNext();
        }
    }
    // 不可达设为-1
    for (auto& d : dist) {
        if (d >= inf)
            d = -1;
    }
    return dist;
}

/// @brief 邻接矩阵O(n^2)
/// @param start
/// @return
std::vector<nodeId_t> MatrixGraph::dijkstra(nodeId_t start)
{
    nodeId_t inf = 0x3f3f3f3f;
    std::vector<nodeId_t> dist(this->vertexNum, inf);
    dist[start] = 0;
    std::vector<bool> vis(vertexNum);
    for (nodeId_t i = 0; i < vertexNum; ++i) {
        nodeId_t min_id = -1;
        for (nodeId_t j = 0; j < vertexNum; ++j) {
            if (!vis[j] && (min_id == -1 || dist[j] < dist[min_id])) {
                min_id = j;
            }
        }
        vis[min_id] = true;
        // 更新距离
        for (nodeId_t j = 0; j < vertexNum; ++j) {
            nodeId_t len = mat(min_id, j);
            len = len > 0 ? len : inf;
            dist[j] = std::min(dist[j], dist[min_id] + len);
        }
    }
    // 不可达设为-1
    for (auto& d : dist) {
        if (d >= inf)
            d = -1;
    }
    return dist;
}

