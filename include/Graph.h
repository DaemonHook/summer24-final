#ifndef GRAPH_H
#define GRAPH_H

#include "Common.h"
#include <cassert>
#include <map>
#include <memory>
#include <utility>
#include <vector>

class LinkGraphNeighborIterator {
public:
    LinkGraphNeighborIterator(long startEdge, long endEdge, std::vector<nodeId_t>* ea, std::vector<weight_t>* weights)
    : _startEdge(startEdge), _endEdge(endEdge)
    {
        _current = startEdge;
        _ea = ea;
        _weights = weights;
    };

    nodeId_t getId() { return _ea->at(_current); }
    weight_t getWeight() { return _weights->at(_current); }

    void toNext()
    {
        _current++;
    }

    bool end() { return _current >= _endEdge; }

public:
    const long _startEdge, _endEdge;
    long _current;
    std::vector<nodeId_t>* _ea;
    std::vector<weight_t>* _weights;
};

/// @brief 图类型的接口
class IGraph {
protected:
    /// @brief 建图的接口（节点编号从0开始）
    /// @param nodeNum 节点个数
    /// @param sources 边的起点
    /// @param dests 边的终点
    /// @param weights 边的权重
    virtual void construct(nodeId_t nodeNum, const std::vector<nodeId_t>& sources,
        const std::vector<nodeId_t>& dests, const std::vector<weight_t>& weights)
        = 0;

public:
    virtual nodeId_t getNodeNum() const = 0;
};

/// @brief 基于紧凑邻接表的图
class LinkGraph : public IGraph {
protected:
    /// @brief 建图的接口（节点编号从0开始）
    /// @param nodeNum 节点个数
    /// @param sources 边的起点
    /// @param dests 边的终点
    /// @param weights 边的权重
    void construct(nodeId_t nodeNum, const std::vector<nodeId_t>& sources,
        const std::vector<nodeId_t>& dests, const std::vector<weight_t>& weights) override;

public:
    LinkGraph(nodeId_t nodeNum, const std::vector<nodeId_t>& sources,
        const std::vector<nodeId_t>& dests, const std::vector<weight_t>& weights)
    {
        vertexNum = nodeNum;
        construct(nodeNum, sources, dests, weights);
    }

    nodeId_t getNodeNum() const { return vertexNum; }

    // 获取节点的后继迭代器
    LinkGraphNeighborIterator getSuccessors(nodeId_t nodeId);

    // va和ea作用见文献
    std::vector<size_t> va;
    std::vector<nodeId_t> ea;
    // 边的权重
    std::vector<weight_t> weights;
    nodeId_t vertexNum;
};

class MatrixGraph : public IGraph {
protected:
    /// @brief 建图的接口（节点编号从0开始）
    /// @param nodeNum 节点个数
    /// @param sources 边的起点
    /// @param dests 边的终点
    /// @param weights 边的权重
    void construct(nodeId_t nodeNum, const std::vector<nodeId_t>& sources,
        const std::vector<nodeId_t>& dests, const std::vector<weight_t>& weights) override;

public:
    MatrixGraph(nodeId_t nodeNum, const std::vector<nodeId_t>& sources,
        const std::vector<nodeId_t>& dests, const std::vector<weight_t>& weights)
    {
        vertexNum = nodeNum;
        construct(nodeNum, sources, dests, weights);
    }

    nodeId_t getNodeNum() const { return vertexNum; }

    weight_t& mat(nodeId_t i, nodeId_t j) { return _mat[i * vertexNum + j]; }

    std::vector<weight_t> _mat;
    nodeId_t vertexNum;
};

#endif