from __future__ import print_function
from __future__ import division
import os
import slpa
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
p.add_argument('--output_file_path', type=str, default='./OUTPUTS',
                help='File path for output')
p.add_argument('--fraction', type=float, default=1, help='option to consider only a fraction of edges')
p.add_argument('--shuffle', action='store_true', help='shuffle the graph data in the file')
p.add_argument('--verbose', action='store_true',
                help='option to print information at regular intervals | not helpful for large graphs')
p.add_argument('--threshold', type=float, default=0.0019, help='threshold for selection of edges')
p.add_argument('--iterations', type=int, default=100, help='number of iterations to propagate for')
p.add_argument('--prob', type=float, default=0.01, help='The probability for soft-clustering')
p = p.parse_args()

assert p.fraction > 0 and p.fraction <= 1, "Fraction limits exceeded"
assert p.threshold >= 0 and p.threshold <= 1, "Threshold limits exceeded"
assert p.prob > 0 and p.prob < 1, "Probability limits exceeded"

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

community = list(slpa.slpa(cur_graph, p.iterations, p.prob))

if verbose:
    print("Final graph has {} communities".format(len(community)))

communityT = {}
for i, set_v in enumerate(community):
    communityT[i] = []
    for v in set_v:
        communityT[i].append(v)

with open(os.path.join(p.output_file_path, 'community_slpa.txt'), 'w') as write_file:
    for i, sets in enumerate(community):
        for v in sets:        
            write_file.write('{}\t{}\n'.format(v, i))

with open(os.path.join(p.output_file_path, 'communityT_slpa.txt'), 'w') as write_file:
    for key in communityT.keys():
        write_file.write('{}'.format(key))
        for persons in communityT[key]:
            write_file.write('\t{}'.format(persons))
        write_file.write('\n')
