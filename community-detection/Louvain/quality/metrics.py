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
p.add_argument('--communities_file', required=True, type=str, help='Root location of the communities information')
p.add_argument('--categoriesT_file', required=True, type=str, help='Root location of the categories T information')
p.add_argument('--categories_file', required=True, type=str, help='Root location of the categories T information')
p.add_argument('--output_file_path', type=str, default='../../../OUTPUTS',
                help='File path for output')
p.add_argument('--num_comm', type=int, default=1000000, help='option to consider only the first num_comm communities')
p.add_argument('--verbose', action='store_true',
                help='option to print information at regular intervals | not helpful for large graphs')
p.add_argument('--natural', action='store_true',
                help='option to pick only natural communities 10 < n < 500')
p = p.parse_args()

catT_filepath = p.categoriesT_file
com_filepath = p.communities_file
cat_filepath = p.categories_file
verbose = p.verbose
natural = p.natural

transpose = {}

# Read Category Transpose file and create dict:
#    PersonID : [Category nums]
with open(catT_filepath, 'r') as graph_file:
    for line in graph_file:
        vals = line.strip().split('\t')

        person_no = vals[0]
        
        transpose[person_no] = vals[1:]

# Read category file and find size of each category        
cat_size = {}        
with open(cat_filepath, 'r') as graph_file:
    for line in graph_file:
        vals = line.strip().split('\t')

        category_no = vals[0]        
        cat_size[category_no] = len(vals) - 2
    
# For each community compute the metrics
with open(com_filepath, 'r') as graph_file:
    count = 0
    tot_f_measure = 0
    tot_precision = 0
    tot_recall = 0
    community_no = 0
    for line in graph_file:
        intersection =  {}
        vals = line.strip().split('\t')

        community_no = vals[0]
        
        if natural and (len(vals) < 12 or len(vals) > 500):
            continue            
        
        for i in range(1,len(vals)):
            if vals[i] in transpose.keys():
                for category in transpose[vals[i]]:
                    if category not in intersection.keys():
                        intersection[category] = 1
                    else:
                        intersection[category] += 1
                    
        # Find precision and recall for each category
        max_f_measure = 0
        max_precision = 0
        max_recall = 0
        for category in intersection:
            precision = intersection[category] / (len(vals) - 1)
            recall = intersection[category] / cat_size[category]
            f_measure = 2. * precision * recall / (precision + recall)
            if f_measure > max_f_measure:
                max_f_measure = f_measure
            if precision > max_precision:
                max_precision = precision
            if recall > max_recall:
                max_recall = recall
            
        tot_f_measure += max_f_measure
        tot_precision += max_precision
        tot_recall += max_recall
            
        if verbose:
            if count % 1000 == 0:
                print("Added {} communities".format(count))

        count += 1
        if count == p.num_comm:
            break
    
    community_no = int(community_no)
    avg_f_measure = tot_f_measure / (count)
    avg_precision = tot_precision / (count)
    avg_recall = tot_recall / (count)

print("Avg f Measure : {}".format(avg_f_measure))
print("Avg Precision : {}".format(avg_precision))
print("Avg Recall : {}".format(avg_recall))
print("Communities : {}".format(count))
