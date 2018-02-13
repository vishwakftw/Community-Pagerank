import os
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
# person_file_root replaces the presence of top_k_sort
# Deliberate addition of this and not use of person.py due to performance issues.
p.add_argument('--person_file_root', type=str, 
                help='Root location of the person information incase person info required in final output')
p.add_argument('--fraction', type=float, default=1, help='option to consider only a fraction of edges')
p.add_argument('--shuffle', action='store_true', help='shuffle the graph data in the file')
p.add_argument('--verbose', action='store_true', help='option to print information at regular intervals')
p.add_argument('--k', type=int, default=25, help='Top-K vals from PageRank')
# p.add_argument('--top_k_sort', action='store_true', help='option to sort the Top K nodes based on PageRank value')
p.add_argument('--threshold', type=float, default=0.0019, help='threshold for selection of edges')
p = p.parse_args()

assert p.fraction > 0 and p.fraction <= 1, "Fraction limits exceeded"
assert p.threshold >= 0 and p.threshold <= 1, "Threshold limits exceeded"

threshold = p.threshold

filepath = p.file_root
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

        if p.verbose:
            if count % 100 == 0:
                print("Added {} edges".format(cur_graph.size()))

        if count == edges_to_add:
            break

if p.verbose:
    print("Graph constructed")

pagerank_result = nx.pagerank(cur_graph)  # run PageRank on the constructed undirected graph without NumPy

pagerank_result = np.array(list(pagerank_result.items()))
nodes, pagerank_vals = np.split(pagerank_result, 2, axis=1)
nodes = nodes.reshape(-1)
pagerank_vals = np.array(pagerank_vals, dtype=float).reshape(-1)

# Get top-k values
top_k_indices = np.argpartition(pagerank_vals, -p.k)[-p.k:]
top_k_vals = pagerank_vals[top_k_indices]
top_k_nodes = nodes[top_k_indices]

# Get the persons info
if p.person_file_root is not None:
    with open(p.person_file_root, 'r') as person_file:
        persons = {}
        for line in person_file:
            vals = line.split('\t')
            vals[4] = vals[4][:-1]
            persons[vals[0]] = vals[1:]

# Print information to file
with open('top-{}-nodes.txt'.format(p.k), 'w') as write_file:
    for n in top_k_nodes:
        write_file.write('{}\n'.format(n))
    if p.verbose:
        print("Top k information saved")

if p.person_file_root is not None:  # Sort and print, if required
    sort_index = np.argsort(top_k_vals)[::-1]
    with open('top-{}-nodes-sorted.txt'.format(p.k), 'w') as write_file:
        write_file.write('Person ID\tName\tGender\tBirth Year\tDeath Year\n')
        for n in top_k_nodes[sort_index]:
            if n in persons.keys():
                personal_info = persons[n]
                write_file.write('{}\t{}\t{}\t{}\t{}\n'.format(n, personal_info[0],
                                 personal_info[1], personal_info[2], personal_info[3]))
            else:
                # WP_1095706 currently prints out as a NA entry.
                # Comment out if this entry is to be ignored
                write_file.write('{}\tNA\tNA\tNA\tNA\n'.format(n))
        if p.verbose:
            print("Top k information saved")
