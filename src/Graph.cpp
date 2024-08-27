#include "Graph.h"
#include <iostream>

void LinkGraph::construct(nodeId_t nodeNum, const std::vector<nodeId_t>& sources,
    const std::vector<nodeId_t>& dests, const std::vector<weight_t>& weights)
{
    va.resize(nodeNum, NO_EDGE);
    inDeg.resize(nodeNum);
    outDeg.resize(nodeNum);

    // 需要 node id 的顺序
    std::map<nodeId_t, std::vector<std::pair<nodeId_t, weight_t>>> links;

    long edgeNum = sources.size();

    assert(sources.size() == dests.size() && dests.size() == weights.size());

    for (long i = 0; i < edgeNum; i++) {
        nodeId_t source = sources[i];
        outDeg[source]++;
        nodeId_t dest = dests[i];
        inDeg[source]++;
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

LinkGraphNeighborIterator LinkGraph::getSuccessors(nodeId_t nodeId)
{
    // 没有后继
    if (va[nodeId] == NO_EDGE) {
        return {0, 0, nullptr, nullptr};
    }

    long startEdge = va[nodeId];
    
    nodeId_t nextNode = nodeId + 1;
    long endEdge;

    // 处理下一个节点没有出边的情况
    while (nextNode != vertexNum && va[nextNode] == NO_EDGE) {
        nextNode++;
    }

    // 如果没有下一个节点，那么end就为ea的最后一个
    if (nextNode == vertexNum) {
        endEdge = (long)ea.size();
    } else {
        endEdge = va[nextNode];
    }
    return LinkGraphNeighborIterator(startEdge, endEdge, &ea, &weights);
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
