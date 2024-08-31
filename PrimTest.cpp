#include "CudaCheckError.h"
#include "CudaGraph.h"
#include "CudaMST.h"
#include "Graph.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <utility>

using namespace std;

const int MAX = 1e6 + 5;
typedef pair<long long, int> PII;
bool marked[MAX];
vector<PII> adj[MAX];
vector<long long int> parent(MAX);

/**
 * Prims Algorithm to find MST using Priority Queue
 **/
long long prim(int x)
{
    priority_queue<PII, vector<PII>, greater<PII>> Q;
    int y;
    long long minimumCost = 0;
    PII p;
    // init
    Q.push(make_pair(0, x));
    parent[0] = -1;
    while (!Q.empty()) {
        // get minimum edge and corresponding node
        p = Q.top();
        Q.pop();
        x = p.second;
        long long int value = p.second;
        // Check if this node is added
        if (marked[x] == true)
            continue;
        // add edge and node to mst
        minimumCost += p.first;
        marked[x] = true;
        // update neighbours for the node
        for (int i = 0; i < adj[x].size(); ++i) {
            y = adj[x][i].second;
            if (marked[y] == false) {
                parent[adj[x][i].second] = value;
                Q.push(adj[x][i]);
            }
        }
    }
    return minimumCost;
}

int main()
{
    nodeId_t nodeNum;
    int edgeNum;
    cin >> nodeNum >> edgeNum;

    vector<vector<pair<int, int>>> adjList(nodeNum);

    nodeId_t first = -1;
    for (int i = 0; i < edgeNum; i++) {
        nodeId_t source, dest;
        weight_t weight;
        cin >> source >> dest >> weight;
        if (first == -1) {
            first = source;
        }
        adjList[source].push_back(make_pair(dest, weight));
        adjList[dest].push_back(make_pair(source, weight));
        adj[source].push_back(make_pair(weight, dest));
        adj[dest].push_back(make_pair(weight, source));
    }

    auto t1 = chrono::steady_clock::now();
    auto res1 = prim(first);
    cout << "MST cpu res: " << res1 << endl;
    auto t2 = chrono::steady_clock::now();
    auto timeUsed = chrono::duration_cast<chrono::milliseconds>(chrono::duration<double>(t2 - t1));
    cout << "MST cpu time: " << timeUsed.count() << endl;

    cudaEvent_t start, stop;
    float duration;
    checkError(cudaEventCreate(&start));
    checkError(cudaEventCreate(&stop));
    checkError(cudaEventRecord(start));
    auto res = cudaPrim(adjList, nodeNum, edgeNum);
    cout << "MST gpu res: " << res1 << endl;
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&duration, start, stop);
    cudaEventDestroy(start);
    cout << "MST gpu time: " << duration << endl;
}