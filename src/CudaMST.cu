#include "CudaMST.h"
#include <ctime>
#include <cuda.h>
#include <iostream>
#include <thrust/device_free.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <vector>
using namespace std;

long long cudaPrim(std::vector<std::vector<std::pair<int, int>>>& adjacency_list, int nodes, int edges)
{
    // create compressed adjancency list
    int* V = new int[nodes];
    int* E = new int[2 * edges];
    int* W = new int[2 * edges];
    int cumulative_sum = 0, limit;

    for (int i = 0; i < nodes; ++i) {
        V[i] = cumulative_sum;
        limit = adjacency_list[i].size();
        for (int j = 0; j < limit; ++j) {
            E[cumulative_sum + j] = adjacency_list[i][j].first;
            W[cumulative_sum + j] = adjacency_list[i][j].second;
        }
        cumulative_sum += limit;
    }

    // ======================== Variables init ====================================
    // sum of edge weights in MST
    long long int edge_sum = 0;
    // current vertex under consideration
    int current = 0;
    // count of vertex added to MST
    int count = 0;

    int* parent = new int[nodes];
    vector<int> weights(nodes);
    bool* inMST = new bool[nodes];
    // init parents, weight and inMST array
    parent[0] = -1;
    for (int i = 0; i < nodes; ++i) {
        weights[i] = INT_MAX;
        inMST[i] = false;
    }

    // device vector for the weights array
    thrust::device_vector<int> device_weights(weights.begin(), weights.end());
    thrust::device_ptr<int> ptr = device_weights.data();

    while (count < nodes - 1) {
        // add current vertex to MST
        ++count;
        inMST[current] = true;

        // update weights and parent arrays as per the current vertex in consideration
        int len = adjacency_list[current].size();
        for (int i = 0; i < len; ++i) {
            int incoming_vertex = adjacency_list[current][i].first;
            if (!inMST[incoming_vertex]) {
                if (weights[incoming_vertex] > adjacency_list[current][i].second) {
                    weights[incoming_vertex] = adjacency_list[current][i].second;
                    parent[incoming_vertex] = current;
                }
            }
        }

        // move/copy the host array to device
        device_weights = weights;

        // get the min index
        int min_index = thrust::min_element(ptr, ptr + nodes) - ptr;
        // cout<<"Min Weight Index: "<<min_index<<endl;

        // add the least edge weight found outto answer
        parent[min_index] = current;
        edge_sum += weights[min_index];
        // reset weight to INT_MAX for this vertex as it is already considered in MST
        weights[min_index] = INT_MAX;
        // new current
        current = min_index;
    }
    free(V);
    free(E);
    free(W);
    free(parent);
    free(inMST);
    return edge_sum;
}
