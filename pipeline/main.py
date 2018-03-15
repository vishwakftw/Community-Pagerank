from __future__ import print_function
from __future__ import division
import os
import numpy as np
import community as c
import networkx as nx
from argparse import ArgumentParser as AP

def _lines_in_file(file_path):
    with open(file_path, 'r') as f:
        for i, _ in enumerate(f):
            pass
    return i + 1

p = AP()
p.add_argument('--file_root', required=True, type=str, help='Root location of the graph information')
p.add_argument('--output_file_path', type=str, default='./OUTPUTS', help='File path for output')
p.add_argument('--fraction', type=float, default=1, help='option to consider only a fraction of edges')
p.add_argument('--shuffle', action='store_true', help='shuffle the graph data in the file')
p.add_argument('--verbose', action='store_true',
                help='option to print information at regular intervals | not helpful for large graphs')
p.add_argument('--threshold', type=float, default=0.0019, help='threshold for selection of edges')
p.add_argument('--k', type=int, default=5, help='value for number of top people')
p = p.parse_args()

assert p.fraction > 0 and p.fraction <= 1, "Fraction limits exceeded"
assert p.threshold >= 0 and p.threshold <= 1, "Threshold limits exceeded"

threshold = p.threshold
filepath = p.file_root
verbose = p.verbose

if p.shuffle:
    os.system('shuf {} -o {}_shuf.txt'.format(filepath, filepath[:-4]))
    filepath = filepath[:-4] + '_shuf.txt'

# Get graph information, graphs can be large hence creating in streaming fashion
cur_graph = nx.Graph()
edges_to_add = int(_lines_in_file(filepath) * p.fraction)
print("Number of edges to add: {}".format(edges_to_add))

with open(filepath, 'r') as graph_file:
    count = 0
    for line in graph_file:
        vals = line.split()
        edge_weight = float(vals[2])

        if edge_weight < threshold:
            continue

        cur_graph.add_edge(vals[1], vals[0], weight=edge_weight)
        count += 1

        if verbose:
            if count % 100 == 0:
                print("Added {} edges".format(cur_graph.size()))

        if count == edges_to_add:
            break

if verbose:
    print("Graph constructed")

partition = c.best_partition(cur_graph)

if verbose:
    print("Final graph has {} vertices".format(len(partition)))

# Transpose this result
partitionT = {}
for key in partition:
    if partition[key] not in partitionT.keys():
        partitionT[partition[key]] = {key}
    else:
        partitionT[partition[key]].add(key)

# Construct community graphs
comm_graphs = []
for n_g in partitionT.keys():
    new_graph = nx.Graph()
    comm_node_list = sorted(list(partitionT[n_g]))
    for i in range(0, len(comm_node_list)):
        for j in range(i + 1, len(comm_node_list)):
            U = comm_node_list[i]
            V = comm_node_list[j]
            try:
                new_graph.add_edge(U, V, weight=cur_graph[U][V]['weight'])
            except KeyError:
                continue

    comm_graphs.append(new_graph)

for G in comm_graphs:
    pagerank_result = nx.pagerank(G)  # run PageRank on the constructed undirected graph without NumPy

    pagerank_result = np.array(list(pagerank_result.items()))
    nodes, pagerank_vals = np.split(pagerank_result, 2, axis=1)
    nodes = nodes.reshape(-1)
    pagerank_vals = np.array(pagerank_vals, dtype=float).reshape(-1)

    # Get top-k values
    top_k_indices = np.argpartition(pagerank_vals, -p.k)[-p.k:]
    top_k_vals = pagerank_vals[top_k_indices]
    top_k_nodes = nodes[top_k_indices]
