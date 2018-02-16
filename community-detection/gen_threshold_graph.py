from argparse import ArgumentParser as AP

p = AP()
p.add_argument('--root', required=True, type=str, help='Root directory for the network file')
p.add_argument('--thres', required=True, type=float, help='Edge weight threshold')
p = p.parse_args()

threshold = p.thres

output = open(p.root[:-4] + '_thres.txt', 'w')

with open(p.root, 'r') as origin:
    for line in origin:
        vals = line.split('\t')
        edge_weight = float(vals[2])
        if edge_weight >= threshold:
            output.write('{}\t{}\t{}\n'.format(vals[0], vals[1], edge_weight))
output.close()
