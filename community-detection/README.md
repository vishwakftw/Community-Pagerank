### Community Detection using SLPA
----------------------------------

+ We make use of the Java archive provided by the authors of the paper **Towards linear time overlapping community detection in social network** by Xie and Szymanski.
+ The usage for the Wikipedia Social Network is 
```java
java -jar SLPA.jar -i <network_file> -Onc 2 -r 0.01 -Sym 1 -seed <seed_number>
```
+ The default seed value used is _1729_.
+ The raw network file is used which can be found in this [link](http://dbs.ifi.uni-heidelberg.de/index.php?id=data).
+ Since graph pre-processing in abstracted away, the only way to perform threshold is externally. To obtain edge thresholded version of the graph, run
```
python3 gen_threshold_graph.py --thres <thres_val> --root <network_file_root>
```
+ Dependencies for running the Java archive is resolved by `commons-collections-3.2.1.jar`. Please don't remove this file from the working directory when running.
