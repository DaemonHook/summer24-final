#include "PageRank.h"
#include "CudaGraph.h"
#include <algorithm>
#include <iostream>
#include <math.h>
#include <vector>

constexpr static float eps = 1e-6;

/// @brief l1 范数
static float norm(const std::vector<float>& vec)
{
    float res = 0.0f;
    for (auto f : vec) {
        res += fabs(f);
    }
    return res;
}

static void init(std::vector<float>& mat, int n)
{
    for (int j = 0; j < n; j++) {
        float sum = 0.0;
        for (int i = 0; i < n; i++) {
            sum += matGet(mat, n, i, j);
        }
        for (int i = 0; i < n; i++) {
            if (fabs(sum) > eps) {
                matGet(mat, n, i, j) /= sum;
            } else {
                matGet(mat, n, i, j) = (1.0f / n);
            }
        }
    }
}

/// @brief 使用幂法求解pagerank
std::vector<float> pageRank(MatrixGraph& graph)
{
    int max_iter = 1000;

    nodeId_t n = graph.getNodeNum();
    std::vector<float> r(n);
    std::vector<float> rLast(n);

    std::vector<float> mat(n * n);

    // 连通为1，不连通为0
    std::transform(graph._mat.begin(), graph._mat.end(), mat.begin(), [](weight_t i) { if (i > 0) return 1; else return 0; });

    init(mat, n);

    for (int i = 0; i < n; i++) {
        r[i] = 1 / (float)n;
    }

    while (max_iter--) {
        for (int i = 0; i < n; i++) {
            rLast[i] = r[i];
        }
        for (int i = 0; i < n; i++) {
            float sum = 0.0;
            for (int j = 0; j < n; j++) {
                sum += rLast[j] * matGet(mat, n, i, j);
            }
            r[i] = sum;
        }
        for (int i = 0; i < n; i++) {
            rLast[i] -= r[i];
        }
        if (norm(rLast) < eps) {
            return r;
        }
    }
    return r;
}