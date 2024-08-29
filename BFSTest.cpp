#include "CudaBFS.h"
#include "CudaGraph.h"
#include "Graph.h"
#include <algorithm>
#include <iostream>

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
    LinkGraph lg(nodeNum, sources, dests, weights);
    // for_each(lg.va.begin(), lg.va.end(), [](int i) { cout << i << ' '; });
    // cout << endl;
    auto res = cudaBFS(lg, 0);
    for_each(res.begin(), res.end(), [](int i) { cout << i << ' '; });
    cout << endl;
}