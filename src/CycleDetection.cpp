#include "Graph.h"

bool LinkGraph::hasCycle()
{
    std::vector<long> inD = inDeg;
    std::queue<nodeId_t> Q;
    for (nodeId_t i = 0; i < inDeg.size(); ++i) {
        if (!inD[i])
            Q.push(i);
    }
    nodeId_t cnt = 0;
    while (!Q.empty()) {
        nodeId_t size = Q.size();
        cnt += size;
        while (size--) {
            nodeId_t front = Q.front();
            Q.pop();
            auto pointer = getSuccessors(front);
            while (!pointer.end()) {
                nodeId_t nei = pointer.getId();
                inD[nei]--;
                if (!inD[nei])
                    Q.push(nei);
                pointer.toNext();
            }
        }
    }
    return cnt != vertexNum;
}

bool MatrixGraph::hasCycle()
{
    return false;
}
