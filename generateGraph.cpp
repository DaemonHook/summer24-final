#include <algorithm>
#include <random>
#include <stdio.h>
#include <string>
#include <unordered_set>
#include <utility>
using namespace std;

using ll = long long;

pair<ll, ll> edgeToCord(ll edge, ll n)
{
    return { edge / n, edge % n };
}

int main(int argc, char** argv)
{
    ll n = atoll(argv[1]);
    ll m = atoll(argv[2]);
    printf("%lld %lld\n", n, m);
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> weightDistr(0, 8172);
    uniform_int_distribution<ll> edgeDistr(0, n * n);
    unordered_set<ll> visited;

    for (ll i = 0; i < m;) {
        ll t = edgeDistr(gen);
        if (!visited.count(t)) {
            visited.insert(t);
            auto [source, dest] = edgeToCord(t, n);
            printf("%lld %lld %d\n", source, dest, weightDistr(gen));
            i++;
        }
    }
}