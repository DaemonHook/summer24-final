"""
使用方法：python generateGraph.py <n> <p>
其中，n是节点数，p是边的概率
权重随机赋予，以[0, 32767]随机分布
输出为算法的输入格式，即第一行为节点数和边数：
n m
以下m行分别为边的起点，终点，权重：
s d w
"""

import random
import networkx as nx
import matplotlib.pyplot as plt
import sys

# 生成一个含有 n 个节点，每条边以 p 的概率出现的随机图
n = int(sys.argv[1])  # 节点数
p = float(sys.argv[2]) # 边的出现概率
G = nx.gnp_random_graph(n, p, directed=True)

for u, v in G.edges():
    G[u][v]["weight"] = random.randint(0, 32767)

print(f'{n} {len(G.edges)}')
for u, v, data in G.edges(data=True):
    print(f"{u} {v} {data['weight']}")