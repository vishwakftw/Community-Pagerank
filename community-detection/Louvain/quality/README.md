#### How good are my communities?
---------------------------------

+ Here we compute the quality of the communities created.
+ First we take the transpose of the Wikipedia graph's edges, for easier computation later. The script that implements this transpose is `transpose.py`. Options for executing this script:
```
python transpose.py [-h] --file_root FILE_ROOT
                    [--output_file_path OUTPUT_FILE_PATH]
                    [--num_comm NUM_COMM] [--verbose]
```
+ After obtaining the "transposed" graph, we can compute the metrics. Metrics computable are **F1-Score**, **Recall** and **Precision**. The script for this purpose is `metrics.py`. Options for executing this script:
```
python metrics.py [-h] --communities_file COMMUNITIES_FILE --categoriesT_file
                  CATEGORIEST_FILE --categories_file CATEGORIES_FILE
                  [--output_file_path OUTPUT_FILE_PATH] [--num_comm NUM_COMM]
                  [--verbose] [--natural]
```
