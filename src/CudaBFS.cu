#include "CudaBFS.h"
#include "CudaCheckError.h"
#include "CudaGraph.h"

__global__ static void init(nodeId_t nodeNum, nodeId_t sourceId, distance_t* d_dist, bool* d_frontier, bool* d_visited)
{
    int tId = blockIdx.x * blockDim.x + threadIdx.x;
    if (tId < nodeNum) {
        d_visited[tId] = false;
        if (tId == sourceId) {
            d_dist[tId] = 0;
            d_frontier[tId] = true;
        }
    }
}

/// @brief 进行一次BFS扩展
__global__ static void doALevel(nodeId_t nodeNum, long edgeNum, bool* toContinue, bool* visited, bool* frontier,
    distance_t* distance, nodeId_t* edgeStartIndices, nodeId_t* edgeEndIndices, nodeId_t* ea)
{
    int tId = blockIdx.x * blockDim.x + threadIdx.x;

    if (tId < nodeNum) {
        // 如果当前node在队列里面
        if (frontier[tId]) {
            // 出队
            frontier[tId] = 0;
            // 设置visited
            visited[tId] = 1;

            // 遍历所有邻居
            nodeId_t start = edgeStartIndices[tId];
            nodeId_t end = edgeEndIndices[tId];
            for (nodeId_t i = start; i < end; i++) {
                nodeId_t neighbor = ea[i];
                if (!visited[neighbor]) {
                    distance[neighbor] = distance[tId] + 1;
                    // 入队
                    frontier[neighbor] = true;
                    *toContinue = true;
                }
            }
        }
    }
}

std::vector<distance_t> cudaBFS(LinkGraph& graph, nodeId_t sourceId)
{
    CudaLinkGraph cudaLG(graph);
    bool* d_frontier;
    bool* d_visited;
    int* d_distance;
    bool* toContinue;

    int block = 1024;
    int grid = ((cudaLG.nodeNum + 1023) / 1024);

    checkError(cudaMalloc(&d_frontier, cudaLG.nodeNum * sizeof(bool)));
    checkError(cudaMalloc(&d_visited, cudaLG.nodeNum * sizeof(bool)));
    checkError(cudaMalloc(&d_distance, cudaLG.nodeNum * sizeof(distance_t)));
    checkError(cudaMallocHost((void**)&toContinue, sizeof(bool)));
    *toContinue = true;
    init<<<grid, block>>>(cudaLG.nodeNum, sourceId, d_distance, d_frontier, d_visited);
    checkError(cudaDeviceSynchronize());

    // int level = 0;
    while (*toContinue) {
        *toContinue = false;
        // printf("doing level %d\n", level++);
        doALevel<<<grid, block>>>(cudaLG.nodeNum, cudaLG.edgeNum, toContinue, d_visited, d_frontier, d_distance,
            cudaLG.d_edgeIndicesStart, cudaLG.d_edgeIndicesEnd, cudaLG.d_weights);
        checkError(cudaDeviceSynchronize());
    }

    std::vector<distance_t> distances(cudaLG.nodeNum);
    checkError(cudaMemcpy(distances.data(), d_distance, cudaLG.nodeNum * sizeof(distance_t), cudaMemcpyDeviceToHost));

    checkError(cudaFree(d_frontier));
    checkError(cudaFree(d_visited));
    checkError(cudaFree(d_distance));
    checkError(cudaFreeHost(toContinue));

    return distances;
}