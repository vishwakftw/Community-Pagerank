### PageRank for Network Centrality
-----------------------------------

+ We make use of [Networkx](https://networkx.github.io).
+ Results can be obtained by running `main.py` using Python3. Available options are:
```bash
python3 main.py [-h] --file_root FILE_ROOT
                [--person_file_root PERSON_FILE_ROOT] [--fraction FRACTION]
                [--shuffle] [--verbose] [--k K] [--threshold THRESHOLD]
```
+ The raw network file is used which can be found at [link](http://dbs.ifi.uni-heidelberg.de/index.php?id=data).
+ Graph pre-processing is done internally i.e., thresholding. The options for the script will show possibilities.
