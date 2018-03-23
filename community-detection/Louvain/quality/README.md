Here we compute the quality of the communities created.

First we take the transpose of the wikipedia's edges, for easier computation later.
Command - python3 transpose.py --file_root=path/wsn_category-person.txt

After obtaining transpose to obtain the metrics
Command - python3 metrics.py --communities_file=path/OUTPUTS/partitionT_multilabel.txt --categoriesT_file=path/OUTPUTS/category_T.tsv --categories_file=path/wikipedia_social_network/wsn_category-person.txt [--natural]
where category_T.csv is the transpose of wikipedia categories created by the previous command
and partitionT_multilabel is the transpose of partition obtained by Louvain's algorithm.
