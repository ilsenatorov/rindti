import sys
import networkx as nx
import numpy as np
from copy import copy


def karger(graph):
    start_edge = np.random.choice(graph.nodes, size=2, replace=False)
    return nx.minimum_cut(graph, start_edge[0], start_edge[1])


def partition(graph, training):
    g = graph.copy()
    if training < 0.5:
        training = 1 - training

    score, new_score = len(g), len(g)
    train, test = set(), set()
    best = train, test
    while True:
        _, (p1, p2) = karger(g)
        max_p = max(p1, p2, key=len)
        min_p = min(p1, p2, key=len)
        train = train.union(max_p).difference(min_p)
        test = test.union(min_p).difference(max_p)

        new_score = (len(train) / len(test)) - (training / (1 - training))
        if new_score > 0:
            g = g.subgraph(train)
        else:
            g = g.subgraph(test)

        new_score = abs(1 - new_score)
        print(new_score, "|", len(train), "|", len(test))
        if new_score < score:
            best = copy(train), copy(test)
            score = new_score
        else:
            break

    return best


def main(dataset):
    graph = nx.Graph()
    with open(dataset, "r") as data:
        for line in data.readlines()[1:]:
            prot, drug = line.split("\t")[:2]
            graph.add_node(prot)
            graph.add_node(drug)
            graph.add_edge(prot, drug, capacity=1)

    partition(graph, 0.7)


if __name__ == '__main__':
    main(sys.argv[1])
