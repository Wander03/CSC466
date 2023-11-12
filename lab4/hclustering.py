"""
Course: CSC 466
Quarter: Fall 2023
Assigment: Lab 4

Name(s):
    Sophia Chung // spchung@calpoly.edu
    Andrew Kerr // adkerr@calpoly.edu

Description:
    How to run: python3 kmeans.py 
                    <input file: consists of data to build classifier> 
                    [OPTIONAL: threshold: threshold at which the program will "cut" the cluster hierarchy to report the clusters]
                    <distance metric: 1 - eucledian, 2 - manhattan, 3 - cosine sim>
                    <min-max standardization: 1 - preform, 0 - do not preform>
"""
import pandas as pd
import numpy as np
import json
from sys import argv
from cluster2 import Cluster
from kmeans import euclidean_dist, manhattan_dist, cosine_sim


def single_link(dm, dist):
    if dist == 3:
        return np.unravel_index(np.argmax(dm), dm.shape)
    return np.unravel_index(np.argmin(dm), dm.shape)

def merge_clusters(clusters, dm, dist):
    if len(clusters) == 1:
        return clusters[0]
    
    merge1, merge2 = single_link(dm, dist)

    merge1_cluster = dm.columns[int(merge1)]
    merge2_cluster = dm.columns[int(merge2)]

    new_cluster = merge1_cluster + ',' + merge2_cluster

    dm[merge1_cluster] = dm[[merge1_cluster, merge2_cluster]].apply(lambda row: min(row[merge1_cluster], row[merge2_cluster]), axis=1)
    dm.loc[merge1_cluster] = dm[merge1_cluster]
    dm.rename(index={merge1_cluster: new_cluster}, columns={merge1_cluster: new_cluster}, inplace=True)
    dm.drop(merge2_cluster, axis=0, inplace=True)
    dm.drop(merge2_cluster, axis=1, inplace=True)

    height = dm[new_cluster].loc[new_cluster]

    if dist == 3:
        dm[new_cluster].loc[new_cluster] = -np.inf
    else:
        dm[new_cluster].loc[new_cluster] = np.inf

    C1 = clusters.index(Cluster(merge1_cluster))
    C2 = clusters.index(Cluster(merge2_cluster))

    clusters.append(Cluster(new_cluster, height, [clusters[C1], clusters[C2]]))
    clusters.remove(Cluster(merge1_cluster))
    clusters.remove(Cluster(merge2_cluster))
    
    return merge_clusters(clusters, dm, dist)

def h_clustering(D, D_filtered, dist, stand):
    if stand:
        # Min-max standardization
        for col, val in D_filtered.items():
            min_val = val.min()
            max_val = val.max()
            D_filtered[col] = (val - min_val) / (max_val - min_val)

    # Compute distance matrix
    if dist == 1:
        dist_matrix = D_filtered.apply(euclidean_dist, args=(D_filtered,True), axis=1)
        np.fill_diagonal(dist_matrix.values, np.inf)
    elif dist == 2:
        dist_matrix = D_filtered.apply(manhattan_dist, args=(D_filtered,True), axis=1)
        np.fill_diagonal(dist_matrix.values, np.inf)
    else:
        dist_matrix = D_filtered.apply(cosine_sim, args=(D_filtered,True), axis=1)
        np.fill_diagonal(dist_matrix.values, -np.inf)

    # Create clusters
    lst_names = [str(c) for c in dist_matrix.columns]
    dist_matrix.columns = [str(col) for col in lst_names]
    dist_matrix.index = [str(col) for col in lst_names]

    cluster_lst = [Cluster(name, 0, None, D.iloc[int(name)].values.tolist()) for name in lst_names]

    return merge_clusters(cluster_lst, dist_matrix, dist)  

def create_dendrogram(node: Cluster):
    children = node.get_children()
    if children is None:
        return {'type': 'leaf', 'height': 0, 'data': node.get_data()}
    return {'type': 'node', 'height': node.get_height(), 'nodes': [create_dendrogram(children[0]), create_dendrogram(children[1])]}

def cut_dendrogram(node: Cluster, threshold: float):
    if node.get_height() > threshold:
        trimmed_clusters = []
        trimmed_clusters.extend(cut_dendrogram(node.get_children()[0], threshold))
        trimmed_clusters.extend(cut_dendrogram(node.get_children()[1], threshold))
    else:
        trimmed_clusters = [node]
    return trimmed_clusters

def main(argv):
    D = pd.read_csv(argv[1], skiprows=[0], header=None, dtype=str)
    restriction = pd.read_csv(argv[1], nrows=1, header=None).iloc[0].to_list()

    # if threshold is included...
    flag = False
    if len(argv) == 5:
        flag = True
        threshold = float(argv[2])
        distance = int(argv[3])
        standardize = bool(int(argv[4]))
    else:
        distance = int(argv[2])
        standardize = bool(int(argv[3]))

    name = argv[1].split("/")[-1] if "/" in argv[1] else argv[1].split("\\")[-1]
    D_filtered =  D.loc[:, map(bool, restriction)].copy()
    D_filtered.columns = range(D_filtered.shape[1])
    D_filtered = D_filtered.astype(float)

    root = h_clustering(D, D_filtered, distance, standardize)
    dendrogram = {'type': 'root', 'height': root.get_height(), 'nodes':[create_dendrogram(root.get_children()[0]), create_dendrogram(root.get_children()[1])]}

    with open(f".\\dendrograms\\{name[:-4]}Dendrogram.json", "w") as f:
        f.write(json.dumps(dendrogram, indent=4))

    if flag:
        D['cluster'] = None
        clusters = cut_dendrogram(root, threshold)
        with open(f".\\results_hclustering\\{name[:-4]}.out.txt", "w") as f:
            f.write(f"Output for python3 {' '.join(argv)}\n\n")

            if distance == 1:
                f.write(f'Distance Metric: Euclidean Distance\n')
            elif distance == 2:
                f.write(f'Distance Metric: Manhattan Distance\n')
            else:
                f.write(f'Distance Metric: Cosine Simularity\n')

            f.write(f'Standardization: {"Min-Max Standardization" if standardize else "None"}\n')
            f.write(f'Threshold: {threshold}\n\n')

            f.write('-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-\n\n')
            for j in range(len(clusters)):
                points = list(map(int, clusters[j].get_points().split(',')))
                center = D_filtered.iloc[points].mean(axis=0).to_list()

                if distance == 1:
                    dists = [euclidean_dist(D_filtered.iloc[x], center) for x in points]
                elif distance == 2:
                    dists = [manhattan_dist(D_filtered.iloc[x], center) for x in points]
                else:
                    dists = [cosine_sim(D_filtered.iloc[x], center) for x in points]

                f.write(f'Cluster {j}:\nCenter: ')
                for i in center:
                    f.write(f'{i},')
                f.write('\n')
                f.write(f'Max Dist. to Center: {np.max(dists)}\n')
                f.write(f'Min Dist. to Center: {np.min(dists)}\n')
                f.write(f'Avg Dist. to Center: {np.mean(dists)}\n')
                f.write(f'SSE for Cluster: {np.sum(np.array(dists)**2)}\n\n')
                f.write(f'{len(points)} Points:\n')

                for x in points:
                    D['cluster'].iloc[x] = j
                    for i in D.drop('cluster', axis=1).iloc[x].to_list():
                        f.write(f'{i},')
                    f.write('\n')

                f.write('\n')
                f.write('-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-\n\n')
    
        D.to_csv(f".\\data_clustered\\hclustering\\{name[:-4]}_clustered.csv", index=False)

if __name__ == "__main__":
    main(argv)