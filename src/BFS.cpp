#include "Graph.h"

std::vector<nodeId_t> LinkGraph::bfs(nodeId_t start)
{
    std::vector<bool> vis(getNodeNum());
    std::queue<nodeId_t> Q;
    std::vector<nodeId_t> result(1);
    Q.push(start);
    nodeId_t level = 0;
    vis[start] = true;
    while (!Q.empty()) {
        int size = Q.size();
        for (int i = 0; i < size; i++) {
            nodeId_t front = Q.front();
            result[front] = level;
            Q.pop();
            auto pointer = getSuccessors(front);
            while (!pointer.end()) {
                nodeId_t nei = pointer.getId();
                if (!vis[nei]) {
                    vis[nei] = true;
                    Q.push(nei);
                    result.push_back(nei);
                }
                pointer.toNext();
            }
        }
        level++;
    }
    return result;
}