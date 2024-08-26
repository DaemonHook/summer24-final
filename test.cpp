#include "Graph.h"
#include <algorithm>
#include <iostream>
using namespace std;

int main()
{
    int n, m;
    cin >> n >> m;

    vector<nodeId_t> sources, dests;
    vector<weight_t> weights;

    for (int i = 0; i < m; i++) {
        int s, d, w;
        cin >> s >> d >> w;
        sources.push_back(s);
        dests.push_back(d);
        weights.push_back(w);
    }

    LinkGraph linkGraph(n, sources, dests, weights);
    cout << "va: ";
    for_each(linkGraph.va.begin(), linkGraph.va.end(), [](int i) { cout << i << ", "; });
    cout << "\n";
    cout << "ea: ";
    for_each(linkGraph.ea.begin(), linkGraph.ea.end(), [](int i) { cout << i << ", "; });
    cout << "\n";
    cout << "weights: ";
    for_each(linkGraph.weights.begin(), linkGraph.weights.end(), [](int i) { cout << i << ", "; });
    cout << endl;
}