#ifndef GRAPH_H
#define GRAPH_H

#include "Common.h"
#include <cassert>
#include <map>
#include <memory>
#include <utility>
#include <vector>

/// @brief 图类型的接口
class IGraph {
public:
    /// @brief 建图的接口（节点编号从0开始）
    /// @param nodeNum 节点个数
    /// @param sources 边的起点
    /// @param dests 边的终点
    /// @param weights 边的权重
    virtual void construct(nodeId_t nodeNum, const std::vector<nodeId_t>& sources,
        const std::vector<nodeId_t>& dests, const std::vector<weight_t>& weights)
        = 0;

    virtual ~IGraph() = default;
};

/// @brief 基于紧凑邻接表的图
class LinkGraph : public IGraph {
public:
    /// @brief 建图的接口（节点编号从0开始）
    /// @param nodeNum 节点个数
    /// @param sources 边的起点
    /// @param dests 边的终点
    /// @param weights 边的权重
    void construct(nodeId_t nodeNum, const std::vector<nodeId_t>& sources,
        const std::vector<nodeId_t>& dests, const std::vector<weight_t>& weights) override;

    LinkGraph(nodeId_t nodeNum, const std::vector<nodeId_t>& sources,
        const std::vector<nodeId_t>& dests, const std::vector<weight_t>& weights);

    // va和ea作用见文献
    std::vector<size_t> va;
    std::vector<nodeId_t> ea;
    // 边的权重
    std::vector<weight_t> weights;
};

#endif