### Our Pipeline
----------------

+ First we detect communities in the graph using Louvain's algorithm. For this we use the `community_pagerank.py` script. This uses [Networkx](https://networkx.github.io), [Community](https://perso.crans.org/aynaud/communities/index.html) primarily.
    + If the communities have been detected using either Louvain's algorithm or any other algorithm for that matter, you can provide the communities in the following format: `<comm_id> [\t person_id]`.
    + You will also need to provide the details of the networks in an adjacency list format as prescribed in the dataset itself.
+ After this, we will obtain the top _K_ users of each "large" community. By default, "large" means communities with size greater than or equal to 50 and _K_ is 45. This is done using PageRank.
+ We then use [Python-Wikipedia](https://pypi.python.org/pypi/wikipedia) to extract the pages of the top people. The script used is `page_data.py`.
+ This is pre-processed using [NLTK](https://www.nltk.org/) using the script `data_preprocess.py`.
+ The pre-processed data is now fed to `classifier.py` for classification task. We use [Scikit-Learn](http://scikit-learn.org/stable/index.html) for the classifiers. You need to provide the splits of data in two files - one for training and one for testing. Every line in each of those files should follow this structure: `<comm_id> [\t person_id]`.
    + At the end you can choose to only see the accuracy or you can only visualize the confusion matrix using [MatplotLib](https://matplotlib.org/) also.

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
                    [--max_files_per_comm MAX_FILES_PER_COMM]
```
+ For `data_preprocess.py`, the options are:
```
python data_preprocess.py [-h] [--dir_root RAW DATA ROOT] [--verbose]
                          [--clean_data EMPTY_DIR_FOR_OUTPUT] [--stem]
```
+ For `classifier.py`, the options are:
```
python classifier.py [-h] [--clean_data_root CLEAN_DATA_ROOT] [--train TRAIN]
                     [--test TEST] [--op OP] [--show_confusion_matrix]
                     [--verbose] [--classifier CLASSIFIER] [--nfeatures NFEATURES]
                     [--save_load]
```
+ For information about the options can be found using the `-h` tag.

#### Obtaining pages using Python-Wikipedia
-------------------------------------------

+ Given a name of a person, it would rather seem easy to extract pages using the API provided.
+ Unfortunately, that was not the case. There were some pages for which either a `DisambiguationError` or a `PageError` was thrown.
+ This issue was resolved in the following steps:
    + Take a person whose name is `NAME`. Perform `wikipedia.page(NAME)`.
        + If exception doesn't arise, then well and good.
        + If exception arises, then we perform `wikipedia.search(NAME)`.
            + Collect all the candidate pages, and keep only those with `NAME` in it.
            + Obtain the pages, and check the category.
            + Use the page with the highest non-zero category intersection between category provided online, and categories in the dataset.
            + If no such page exists, then we leave that person.
+ This is the jist of what happens [here](https://github.com/vishwakftw/CS6670-TDM/blob/master/pipeline/page_data.py#L93-L115).

#### Classifiers used
---------------------

+ Multinomial Naive-Bayes
+ Multinomial Logistic Regression
+ Multilayer perceptrons with 1 and 2 hidden layers
+ K-Nearest Neighbours
+ One-vs-One classification using Support Vector Machines and Logistic Regression
+ One-vs-Rest classification using Support Vector Machines and Logistic Regression
