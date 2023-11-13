Course: CSC 466
Quarter: Fall 2023
Assignment: Lab 4

Name(s):
    Sophia Chung // spchung@calpoly.edu
    Andrew Kerr // adkerr@calpoly.edu

Output Files:
    kmeans: results_kmeans
    hclustering: results_hclustering, dendrograms
    KNN: results_dbscan

    data_clustered: contains csv files of each data set with and added "cluster" column

Misc Files:
    cluster.py and cluster2.py: Cluster objects used in kmeans and hclustering respectively
    visualizations.Rmd and viz.Rmd: contains code used to create visualizations for report

How to run kmeans.py: 
    python3 kmeans.py 
                <input file: consists of data to build classifier> 
                <K: number of neighbors to consider> 
                <initial centroid selection: 0 - random, 1 - kmeans++>
                <distance metric: 1 - euclidean, 2 - manhattan, 3 - cosine sim>
                <min-max standardization: 1 - perform, 0 - do not perform>
                <stoppage threshold: min (or max for cosine sim) movement of centroids before stopping>

How to run hclustering.py: 
    python3 hclustering.py 
                <input file: consists of data to build classifier> 
                [OPTIONAL: threshold: threshold at which the program will "cut" the cluster hierarchy to report the clusters]
                <distance metric: 1 - euclidean, 2 - manhattan, 3 - cosine sim>
                <min-max standardization: 1 - perform, 0 - do not perform>

How to run dbscan.py:
    python3 dbscan.py 
                <input file: consists of data to build classifier>
                <epsilon: radius in which data points are considered to be neighbors>
                <min_points: number of neighbors required for a data point to be a core point>
                <distance metric: 1 - euclidean, 2 - manhattan, 3 - cosine sim>
                <min-max standardization: 1 - perform, 0 - do not perform>
