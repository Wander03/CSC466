"""
Course: CSC 466
Quarter: Fall 2023
Assigment: Lab 4

Name(s):
    Sophia Chung // spchung@calpoly.edu
    Andrew Kerr // adkerr@calpoly.edu

Description:
    How to run: python3 dbscan.py 
                    <input file: consists of data to build classifier>
                    <epsilon: radius in which data points are considered to be neighbors>
                    <min_points: number of neighbors required for a data point to be a core point>
                    <distance metric: 1 - euclidean, 2 - manhattan, 3 - cosine sim>
                    <min-max standardization: 1 - perform, 0 - do not perform>
"""
import pandas as pd
import numpy as np
from sys import argv
from kmeans import euclidean_dist, manhattan_dist, cosine_sim


class DBSCANClustering():
    def __init__(self, D_filtered, epsilon, min_points, distance, standardize): 
        self.D_filtered = D_filtered
        self.epsilon = epsilon
        self.min_points = min_points
        self.distance = distance
        self.standardize = standardize
        
    def densityConnected(self, point, dist_matrix, core, cluster_id, point_cluster):
        distances = dist_matrix.iloc[point]
        neighbors = np.where(distances <= self.epsilon)[0]
        for d in neighbors:
            if point_cluster[d] == 0: # check if already classified
                point_cluster[d] = cluster_id
                if d in core:
                    self.densityConnected(d, dist_matrix, core, cluster_id, point_cluster)
            
    def dbscan(self):
        if self.standardize:
        # Min-max standardization
            for col, val in self.D_filtered.items():
                min_val = val.min()
                max_val = val.max()
                self.D_filtered[col] = (val - min_val) / (max_val - min_val)

        # Compute distance matrix
        if self.distance == 1:
            dist_matrix = self.D_filtered.apply(euclidean_dist, args=(self.D_filtered, True), axis=1)
            np.fill_diagonal(dist_matrix.values, np.inf)
        elif self.distance == 2:
            dist_matrix = self.D_filtered.apply(manhattan_dist, args=(self.D_filtered, True), axis=1)
            np.fill_diagonal(dist_matrix.values, np.inf)
        else:
            dist_matrix = self.D_filtered.apply(cosine_sim, args=(self.D_filtered, True), axis=1)
            np.fill_diagonal(dist_matrix.values, -np.inf)
        
        # Core point discovery
        core = []
        for i in range(len(self.D_filtered)):
            distances = dist_matrix.iloc[i]
            neighbors = np.where(distances <= self.epsilon)[0]
            if len(neighbors) >= self.min_points:
                core.append(i)

        # Cluster construction
        curr_cluster = 0
        point_cluster = dict.fromkeys(list(range(len(self.D_filtered))), 0)
        for d in core:
            if point_cluster[d] == 0:
                curr_cluster += 1
                point_cluster[d] = curr_cluster
                self.densityConnected(d, dist_matrix, core, curr_cluster, point_cluster) # find all density connected points
                
        cluster = dict.fromkeys(list(range(1, curr_cluster + 1)), [])
        for k in range(1, curr_cluster + 1):
            cluster[k] = [key for key, val in point_cluster.items() if val == k]

        noise = [key for key, val in point_cluster.items() if val == 0]
        border = list(set(range(len(self.D_filtered))) - set(noise) - set(core))
        return cluster, core, noise, border

def main(argv):
    D = pd.read_csv(argv[1], skiprows=[0], header=None, dtype=str)
    restriction = pd.read_csv(argv[1], nrows=1, header=None).iloc[0].to_list()
    epsilon = float(argv[2])
    min_points = int(argv[3])
    distance = int(argv[4])
    standardize = bool(int(argv[5]))

    name = argv[1].split("/")[-1] if "/" in argv[1] else argv[1].split("\\")[-1]
    D_filtered =  D.loc[:, map(bool, restriction)].copy()
    D_filtered.columns = range(D_filtered.shape[1])
    D_filtered = D_filtered.astype(float)

    clustering = DBSCANClustering(D_filtered, epsilon, min_points, distance, standardize)
    res = clustering.dbscan()
    clusters = res[0]
   
    D['cluster'] = None
    # with open(f".\\results_dbscan\\{name[:-4]}.out.txt", "w") as f:
    with open(f"./results_dbscan/{name[:-4]}.out.txt", "w") as f:
        f.write(f"Output for python3 {' '.join(argv)}\n\n")

        if distance == 1:
            f.write(f'Distance Metric: Euclidean Distance\n')
        elif distance == 2:
            f.write(f'Distance Metric: Manhattan Distance\n')
        else:
            f.write(f'Distance Metric: Cosine Similarity\n')

        f.write(f'Standardization: {"Min-Max" if standardize else "None"}\n')

        f.write('-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-\n\n')
        for cluster, points in clusters.items():
            center = D_filtered.iloc[points].mean(axis=0).to_list()

            if distance == 1:
                dists = [euclidean_dist(D_filtered.iloc[x], center) for x in points]
            elif distance == 2:
                dists = [manhattan_dist(D_filtered.iloc[x], center) for x in points]
            else:
                dists = [cosine_sim(D_filtered.iloc[x], center) for x in points]

            f.write(f'Cluster {cluster}:\nCenter: ')
            for i in center:
                f.write(f'{i},')
            f.write('\n')
            f.write(f'Max Dist. to Center: {np.max(dists)}\n')
            f.write(f'Min Dist. to Center: {np.min(dists)}\n')
            f.write(f'Avg Dist. to Center: {np.mean(dists)}\n')
            f.write(f'SSE for Cluster: {np.sum(np.array(dists)**2)}\n\n')
            f.write(f'{len(points)} Points:\n')

            for x in points:
                D['cluster'].iloc[x] = cluster
                for i in D.drop('cluster', axis=1).iloc[x].to_list():
                    f.write(f'{i},')
                f.write('\n')

            f.write('\n')
            f.write('-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-\n\n')

    # D.to_csv(f".\\data_clustered\\dbscan\\{name[:-4]}_clustered.csv", index=False)
    D.to_csv(f"./data_clustered/dbscan/{name[:-4]}_clustered.csv", index=False)

if __name__ == "__main__":
    main(argv)


# python3 dbscan.py data/4clusters.csv 0.1 2 1 1
# python3 dbscan.py data/AccidentsSet03.csv 0.4 3 1 1
# python3 dbscan.py data/iris.csv 0.12 6 1 1
# python3 dbscan.py data/mammal_milk.csv 0.3 3 1 1
# python3 dbscan.py data/planets.csv 0.2 2 1 1