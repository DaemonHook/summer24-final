#include "PageRank.h"
#include "CudaPageRank.h"
#include "Graph.h"
#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;

int main()
{
    nodeId_t nodeNum;
    int edgeNum;
    cin >> nodeNum >> edgeNum;
    vector<nodeId_t> sources, dests;
    vector<weight_t> weights;
    for (int i = 0; i < edgeNum; i++) {
        nodeId_t source, dest;
        weight_t weight;
        cin >> source >> dest >> weight;
        sources.push_back(source);
        dests.push_back(dest);
        weights.push_back(weight);
    }
    MatrixGraph matGraph(nodeNum, sources, dests, weights);
    vector<float> rank = pageRank(matGraph);
    for_each(rank.begin(), rank.end(), [](float f) { cout << f << ' '; });
    cout << endl;
    vector<float> cudaRank = cudaPageRank(matGraph);
    for_each(cudaRank.begin(), cudaRank.end(), [](float f) { cout << f << ' '; });
    cout << endl;
}