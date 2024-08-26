#ifndef GRAPH_H
#define GRAPH_H

#include "Common.h"
#include <cassert>
#include <map>
#include <memory>
#include <utility>
#include <vector>

class NodeIterator {
    virtual nodeId_t get() = 0;
    virtual void toNext() = 0;
    virtual bool end() = 0;
};

/// @brief 图类型的接口
class IGraph {
    /// @brief 建图的接口（节点编号从0开始）
    /// @param nodeNum 节点个数
    /// @param sources 边的起点
    /// @param dests 边的终点
    /// @param weights 边的权重
    virtual void construct(nodeId_t nodeNum, const std::vector<nodeId_t>& sources,
        const std::vector<nodeId_t>& dests, const std::vector<weight_t>& weights)
        = 0;

public:
    // virtual std::unique_ptr<NodeIterator> getNeighbors(nodeId_t nodeId) = 0;

    virtual nodeId_t getNodeNum() const = 0;
};

/// @brief 基于紧凑邻接表的图
class LinkGraph : public IGraph {
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

    // va和ea作用见文献
    std::vector<size_t> va;
    std::vector<nodeId_t> ea;
    // 边的权重
    std::vector<weight_t> weights;
    nodeId_t vertexNum;
};

// class LinkGraphNodeIterator: NodeIterator {
//
// };

class MatrixGraph : public IGraph {
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