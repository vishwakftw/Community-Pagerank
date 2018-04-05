### Our Pipeline
----------------

+ First we detect communities in the graph using Louvain's algorithm. For this we use the `community_pagerank.py` script. This uses [Networkx](https://networkx.github.io), [Community](https://perso.crans.org/aynaud/communities/index.html) primarly.
    + If the communities have been detected using either Louvain's algorithm or any other algorithm for that matter, you can provide the communities in the following format: `<comm_id> [\t person_id]`.
    + You will also need to provide the details of the networks in an adjacency list format as prescribed in the dataset itself.
+ After this, we will obtain the top _K_ users of each "large" community. By default, "large" means communities with size greater than or equal to 50 and _K_ is 45. This is done using PageRank.
+ We then use [Python-Wikipedia](https://pypi.python.org/pypi/wikipedia) to extract the pages of the top people. The script used is `page_data.py`.
+ This is pre-processed using [NLTK](https://www.nltk.org/) using the script `data_preprocess.py`.

#### Instructions for using the code
------------------------------------

+ For `community_pagerank.py`, the options are:
```
python community_pagerank.py [-h] --file_root FILE_ROOT [--fraction FRACTION]
                             [--shuffle] [--verbose] [--threshold THRESHOLD]
                             [--k K] [--min_comm_size MIN_COMM_SIZE]
                             [--comm_src COMM_SRC]
```
+ For `page_data.py`, the options are:
```
python page_data.py [-h] [--root ROOT] --communities COMMUNITIES
                    [--basic_info BASIC_INFO] [--category_info CATEGORY_INFO]
```

+ For `data_preprocess.py`, the options are:
```
python data_preprocess.py [-h] [--dir_root RAW DATA ROOT] [--verbose]
                          [--clean_data EMPTY_DIR_FOR_OUTPUT]
                          
+ For information about the options can be found using the `-h` tag.
