import numpy as np
import networkx as nx
from collections import defaultdict

def slpa(graph, iterations, probability):
    """
    Implements Speaker-Listener Label Propagation Algorithm by Xie and Szymanski
    in their paper Towards linear time overlapping community detection in social network

    Args:
        graph       : Networkx Graph instance
        iterations  : Integer with number of iterations to propagate labels
        probability : Minimum probability of retaining a label

    Returns:
        communities : A list of the form [<community number>: <nodes>+]
    """
    # Set a memory for each node. Step 1
    # Memory for a node is a dictionary: {node_no: number of time seen}
    memories = {}
    for n in graph.nodes:
        memories[n] = {n: 1}

    # Evolve over iterations. Step 2
    for t in range(0, iterations):
        listeners = np.random.permutation(list(graph.nodes))

        for l in listeners:
            speakers = list(graph[l].keys())
            if len(speakers) == 0:
                continue

            label_list = defaultdict(int)

            for sid, s in enumerate(speakers):
                # Speaker suggests a value
                all_occurences = sum(memories[s].values())
                probs = np.random.multinomial(1, [freq / all_occurences for freq in memories[s].values()]).argmax()
                label_list[list(memories[s].keys())[probs]] += 1

            # Listener selects most occuring label
            w = max(label_list, key=label_list.get)

            # Update Listener memory
            if w in memories[l]:
                memories[l][w] += 1
            else:
                memories[l][w] = 1

    # Post-processing, remove low probability labels. Step 3
    del_dicts = {}
    for node, mem in list(memories.items()):
        for label, freq in mem.items():
            if freq / (iterations + 1) < probability:
                del_dicts[node] = label

    for ddn, ddl in list(del_dicts.items()):
        del memories[ddn][ddl]

    # Build communities
    communities = {}
    for node, mem in list(memories.items()):
        for label in mem.keys():
            if label in communities:
                communities[label].add(node)
            else:
                communities[label] = set([node])

    # Remove nested communities
    nested_communities = set()
    keys = communities.keys()
    for i, l1 in enumerate(list(keys)[:-1]):
        comm_comp_1 = communities[l1]
        for l2 in list(keys)[i+1:]:
            comm_comp_2 = communities[l2]
            if comm_comp_1.issubset(comm_comp_2):
                nested_communities.add(l1)
            elif comm_comp_1.issuperset(comm_comp_2):
                nested_communities.add(l2)

    for comm in nested_communities:
        del communities[comm]

    return communities
