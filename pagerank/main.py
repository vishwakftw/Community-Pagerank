import numpy as np
import networkx as nx
from argparse import ArgumentParser as AP

def _lines_in_file(file_path):
    with open(file_path, 'r') as f:
        for i, _ in enumerate(f):
            pass
    return i + 1

p = AP()
p.add_argument('--file_root', required=True, type=str, help='Root location of the graph information')
p.add_argument('--fraction', type=float, default=1, help='option to consider only a fraction of edges')
p.add_argument('--verbose', action='store_true', help='option to print information at regular intervals')
p.add_argument('--k', type=int, default=25, help='Top-K vals from PageRank')
p.add_argument('--top_k_sort', action='store_true', help='option to sort the Top K nodes based on PageRank value')
p = p.parse_args()

assert p.fraction > 0 and p.fraction <= 1, "Fraction limits exceeded"

# Get graph information, graphs can be large hence creating in streaming fashion
cur_graph = nx.Graph()
edges_to_add = int(_lines_in_file(p.file_root) * p.fraction)
print("Number of edges to add: {}".format(edges_to_add))

with open(p.file_root, 'r') as graph_file:
    for line in graph_file:
        vals = line.split()
        cur_graph.add_edge(vals[1], vals[0], weight=float(vals[2]))

        if p.verbose:
            if cur_graph.size() % 100 == 0:
                print("Added {} edges".format(cur_graph.size()))

        edges_to_add -= 1
        if edges_to_add == 0:
            break

if p.verbose:
    print("Graph constructed")

pagerank_result = nx.pagerank_numpy(cur_graph)  # run PageRank on the constructed undirected graph
pagerank_result = np.array(list(pagerank_result.items()))
nodes, pagerank_vals = np.split(pagerank_result, 2, axis=1)
nodes = nodes.reshape(-1)
pagerank_vals = np.array(pagerank_vals, dtype=float).reshape(-1)

# Get top-k values
top_k_indices = np.argpartition(pagerank_vals, -p.k)[-p.k:]
top_k_vals = pagerank_vals[top_k_indices]
top_k_nodes = nodes[top_k_indices]

# Print information to file
with open('top-{}-nodes.txt'.format(p.k), 'w') as write_file:
    for n in top_k_nodes:
        write_file.write('{}\n'.format(n))
    if p.verbose:
        print("Top k information saved")

if p.top_k_sort:  # Sort and print, if required
    sort_index = np.argsort(top_k_vals)[::-1]
    with open('top-{}-nodes-sorted.txt'.format(p.k), 'w') as write_file:
        for n in top_k_nodes:
            write_file.write('{}\n'.format(n))
        if p.verbose:
            print("Top k information saved")
