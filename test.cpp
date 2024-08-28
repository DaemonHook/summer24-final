#include "CudaBFS.h"
#include "CudaGraph.h"
#include "CudaSSSP.h"
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
    LinkGraph linkGraph(nodeNum, sources, dests, weights);
    cout << "va: ";
    for_each(linkGraph.va.begin(), linkGraph.va.end(), [](int i) { cout << i << ' '; });
    cout << endl;
    cout << "ea: ";
    for_each(linkGraph.ea.begin(), linkGraph.ea.end(), [](int i) { cout << i << ' '; });
    cout << endl;

    auto costs = cudaSSSP(linkGraph, 0);
    cout << "costs: ";
    for_each(costs.begin(), costs.end(), [](int i) { cout << i << ' '; });
    cout << endl;
    return 0;
}
