import numpy as np
from matplotlib import pyplot as plt

FILE_NAME = 'community-people-table.txt'

counts = np.empty(0)
with open(FILE_NAME, 'r') as f:
    for line in f:
        counts = np.append(counts, len(line.split('\t')) - 1)

unique_vals = np.unique(counts)
unique_vals = np.sort(unique_vals)
unique_vals = unique_vals[::-1]
print("Top 10 largest communities detected:\n{}".format(unique_vals[:10]))
print("Top 10 smallest communities detected: \n{}".format(unique_vals[-10:]))
print("Number of distinct community sizes: \n{}".format(len(unique_vals)))

value_counts = np.empty(0)
for u in unique_vals:
    value_counts = np.append(value_counts, len(counts[counts == u]))

plt.figure(figsize=(12, 10))
plt.xlabel('Community sizes')
plt.ylabel('Number of communities')
plt.plot(unique_vals, value_counts, 'b', marker='o', markersize=5.0)
plt.semilogy()  # Use log scale for y due to skew
plt.semilogx()  # Use log scale for x due to skew, again
plt.savefig('community-counts.png', dpi=150)
