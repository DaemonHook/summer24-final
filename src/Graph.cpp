#include "Graph.h"

void LinkGraph::construct(nodeId_t nodeNum, const std::vector<nodeId_t>& sources,
    const std::vector<nodeId_t>& dests, const std::vector<weight_t>& weights)
{
    va.resize(nodeNum, NO_EDGE);

    // 需要 node id 的顺序
    std::map<nodeId_t, std::vector<std::pair<nodeId_t, weight_t>>> links;

    long edgeNum = sources.size();

    assert(sources.size() == dests.size() && dests.size() == weights.size());

    for (long i = 0; i < edgeNum; i++) {
        nodeId_t source = sources[i];
        nodeId_t dest = dests[i];
        weight_t weight = weights[i];
        links[source].push_back({ dest, weight });
    }

    for (auto& item : links) {
        nodeId_t curNode = item.first;
        va[curNode] = ea.size();
        for (auto& p : item.second) {
            ea.push_back(p.first);
            this->weights.push_back(p.second);
        }
    }
}

LinkGraph::LinkGraph(nodeId_t nodeNum, const std::vector<nodeId_t>& sources, const std::vector<nodeId_t>& dests, const std::vector<weight_t>& weights)
{
    this->construct(nodeNum, sources, dests, weights);
}

void MatrixGraph::construct(nodeId_t nodeNum, const std::vector<nodeId_t>& sources, const std::vector<nodeId_t>& dests, const std::vector<weight_t>& weights)
{
    _mat.resize(nodeNum * nodeNum, -1);

    assert(sources.size() == dests.size() && dests.size() == weights.size());
    
    long edgeNum = sources.size();

    for (long i = 0; i < edgeNum; i++) {
        nodeId_t source = sources[i];
        nodeId_t dest = dests[i];
        weight_t weight = weights[i];
        mat(source, dest) = weight;
    }
}
