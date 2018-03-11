### Community Detection using Multi-label Propagation
-----------------------------------

+ We make use of [Networkx](https://networkx.github.io) and [Community](https://perso.crans.org/aynaud/communities/index.html).
+ Results can be obtained by running `main.py` using Python3. Available options are:
```bash
python3 main.py [-h] --file_root FILE_ROOT
                [--output_file_path OUTPUT_FILE_PATH] [--fraction FRACTION]
                [--shuffle] [--verbose] [--threshold THRESHOLD]
```
+ The raw network file is used which can be found at [link](http://dbs.ifi.uni-heidelberg.de/index.php?id=data).
+ Graph pre-processing is done internally i.e., thresholding. The options for the script will show possibilities.
