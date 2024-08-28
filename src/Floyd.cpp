#include "Graph.h"

std::vector<std::vector<nodeId_t>> LinkGraph::floyd()
{
    nodeId_t n = this->getNodeNum();
    weight_t inf = 0x3f3f3f3f;
    std::vector<std::vector<nodeId_t>> dist(n, std::vector<int>(n, inf));
    for (int i = 0; i < n; i++) {
        auto pointer = getSuccessors(i);
        while (!pointer.end()) {
            nodeId_t nei = pointer.getId();
            dist[i][nei] = pointer.getWeight();
            pointer.toNext();
        }
    }

    for (int k = 0; k < n; k++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (dist[i][k] != inf
                    && dist[k][j] != inf
                    && dist[i][k] + dist[k][j] < dist[i][j]) {
                    dist[i][j] = dist[i][k] + dist[k][j];
                }
            }
        }
    }

    return dist;
}

std::vector<std::vector<nodeId_t>> MatrixGraph::floyd()
{
    nodeId_t n = this->getNodeNum();
    weight_t inf = 0x3f3f3f3f;
    std::vector<std::vector<nodeId_t>> dist(n, std::vector<int>(n, inf));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            dist[i][j] = mat(i, j);
        }
    }

    for (int k = 0; k < n; k++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (dist[i][k] != inf
                    && dist[k][j] != inf
                    && dist[i][k] + dist[k][j] < dist[i][j]) {
                    dist[i][j] = dist[i][k] + dist[k][j];
                }
            }
        }
    }

    return dist;
}
