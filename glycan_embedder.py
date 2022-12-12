import sys
import glyles


for glycan_file in sys.argv[1:]:
    graphs = []
    with open(glycan_file, "r") as glycans:
        for line in glycans.readlines():
            idx, iupac = line.strip().split("\t")[:2]
            tree = glyles.Glycan(iupac, tree_only=True).get_tree()

            nodes, edges = {}, {}
            for node in tree.nodes:
                nodes[node] = tree.nodes[node].get_name()
                edges[node] = []

            for edge in tree.edges():
                nodes[edge[0]].append(edge[1])
                nodes[edge[1]].append(edge[0])

            graphs.append(grakel.Graph(edges, node_labels=nodes))
