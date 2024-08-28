#include "Graph.h"

std::vector<nodeId_t> LinkGraph::bfs(nodeId_t start)
{
    std::vector<bool> vis(this->getNodeNum());
    std::queue<nodeId_t> Q;
    std::vector<nodeId_t> result;
    Q.push(start);
    result.push_back(start);
    vis[start] = true;
    while (!Q.empty()) {
        nodeId_t front = Q.front();
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
    return result;
}
