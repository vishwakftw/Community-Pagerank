from __future__ import print_function
from __future__ import division
import os
from argparse import ArgumentParser as AP

def _lines_in_file(file_path):
    with open(file_path, 'r') as f:
        for i, _ in enumerate(f):
            pass
    return i + 1

p = AP()
p.add_argument('--file_root', required=True, type=str, help='Root location of the communities information')
p.add_argument('--output_file_path', type=str, default='../../../OUTPUTS',
                help='File path for output')
p.add_argument('--num_comm', type=int, default=1000000, help='option to consider only the first num_comm communities')
p.add_argument('--verbose', action='store_true',
                help='option to print information at regular intervals | not helpful for large graphs')
p = p.parse_args()

filepath = p.file_root
verbose = p.verbose

# Get graph information, graphs can be large hence creating in streaming fashion
# Create the transpose dictionary along the way
communities_to_add = min(_lines_in_file(filepath), p.num_comm)
print("Number of communities to add: {}".format(communities_to_add))

transpose = {}

with open(filepath, 'r') as graph_file:
    count = 0
    for line in graph_file:
        vals = line.strip().split('\t')

        community_no = vals[0]
        
        for i in range(2,len(vals)):
            if vals[i] not in transpose.keys() and vals[i] != '':
                transpose[vals[i]] = [community_no]
            else:
                transpose[vals[i]].append(community_no)
                
        if verbose:
            if count % 10000 == 0:
                print("Added {} communities".format(count))

        count += 1
        if count == p.num_comm:
            break

if verbose:
    print("Dictionary constructed")

with open(os.path.join(p.output_file_path, 'communities_T.tsv'), 'w') as write_file:
    for key in transpose:
        write_file.write('{}'.format(key))
        for communities in transpose[key]:
            write_file.write('\t{}'.format(communities))
        write_file.write('\n')
