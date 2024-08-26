#ifndef GRAPH_H
#define GRAPH_H

#include "Common.h"
#include <cassert>
#include <map>
#include <memory>
#include <utility>
#include <vector>

class NodeIterator {
public:
    virtual nodeId_t get() = 0;
    virtual void toNext() = 0;
    virtual bool end() = 0;
};

class LinkGraphNodeIterator : public NodeIterator {
public:
    LinkGraphNodeIterator(nodeId_t start, nodeId_t end, std::vector<nodeId_t>* ea)
    {
        _start = start;
        _end = end;
        _ea = ea;
        _current = start;
    };

    nodeId_t get() override { return _ea->at(_current); }

    void toNext() override
    {
        _current++;
        assert(_current <= _end);
    }

    bool end() override { return _current == _end; }

private:
    nodeId_t _current, _start, _end;
    std::vector<nodeId_t>* _ea;
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

    std::unique_ptr<NodeIterator> getNeighbors(nodeId_t nodeId)
    {
        nodeId_t start = va[nodeId];
        // if (start == NO_EDGE) {
        //     return std::make_unique<LinkGraphNodeIterator>(0, 0, nullptr);
        // }
        nodeId_t end = nodeId == vertexNum - 1 ? ea.size() : va[nodeId + 1];
        return std::make_unique<LinkGraphNodeIterator>(start, end, &ea);
    }

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