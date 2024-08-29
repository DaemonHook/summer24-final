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
    MatrixGraph matGraph(nodeNum, sources, dests, weights);
    for (int i = 0; i < matGraph.vertexNum * matGraph.vertexNum; i++) {
        cout << i << '\t';
        if (i != 0 && i % matGraph.vertexNum == 0) {
            cout << '\n';
        }
    }
    
    return 0;
}
