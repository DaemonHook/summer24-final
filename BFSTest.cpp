#include "CudaBFS.h"
#include "CudaCheckError.h"
#include "CudaGraph.h"
#include "Graph.h"
#include <algorithm>
#include <chrono>
#include <iostream>

using namespace std;

int main()
{
    nodeId_t nodeNum;
    int edgeNum;
    cin >> nodeNum >> edgeNum;
    vector<nodeId_t> sources, dests;
    vector<weight_t> weights;
    nodeId_t first = -1;
    for (int i = 0; i < edgeNum; i++) {
        nodeId_t source, dest;
        weight_t weight;
        cin >> source >> dest >> weight;
        if (first == -1) {
            first = source;
        }
        sources.push_back(source);
        dests.push_back(dest);
        weights.push_back(weight);
    }
    LinkGraph lg(nodeNum, sources, dests, weights);

    auto t1 = chrono::steady_clock::now();
    auto res1 = lg.bfs(0);
    auto t2 = chrono::steady_clock::now();
    auto timeUsed = chrono::duration_cast<chrono::milliseconds>(chrono::duration<double>(t2 - t1));
    cout << "BFS cpu time: " << timeUsed.count() << endl;

    cudaEvent_t start, stop;
    float duration;
    checkError(cudaEventCreate(&start));
    checkError(cudaEventCreate(&stop));
    checkError(cudaEventRecord(start));
    auto res = cudaBFS(lg, 0);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&duration, start, stop);
    cudaEventDestroy(start);
    cout << "BFS gpu time: " << duration << endl;
}