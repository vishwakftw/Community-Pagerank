### Community Detection using SLPA
----------------------------------

+ We make use of the Java archive provided by the authors of the paper **Towards linear time overlapping community detection in social network** by Xie and Szymanski. [Link](https://sites.google.com/site/communitydetectionslpa/)
+ The usage for the Wikipedia Social Network is 
```java
java -Xmx<size>g -jar SLPA.jar -i <network_file> -Onc 2 -r 0.01 -Sym 1 -seed <seed_number> -loopfactor <loop_factor>
```
+ The default seed value used is _1729_.
+ Considering the size of the network, we use `loop_factor` as suggested, and the default value is _0.2_.
+ The raw network file is used which can be found in this [link](http://dbs.ifi.uni-heidelberg.de/index.php?id=data).
+ Since graph pre-processing in abstracted away, the only way to perform threshold is externally. To obtain edge thresholded version of the graph, run
```
python3 gen_threshold_graph.py --thres <thres_val> --root <network_file_root>
```
+ Dependencies for running the Java archive is resolved by `commons-collections-3.2.1.jar`. Please don't remove this file from the working directory when running.
+ Considering the size of the graph that this algorithm is run over, we increase the heap size using a JVM setting. The default value is 32 for our trials. Please check with your system specifications before running.

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
