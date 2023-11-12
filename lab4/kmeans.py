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
                    <K: number of neighbors to consider> 
                    <initial centroid selection: 0 - random, 1 - Kmeans++>
                    <distance metric: 1 - eucledian, 2 - manhattan, 3 - cosine sim>
                    <min-max standardization: 1 - preform, 0 - do not preform>
                    <stoppage threshold: min (or max for cosine sim) movement of centroids before stopping>
"""
import pandas as pd
import numpy as np
from sys import argv
from cluster import Cluster


def euclidean_dist(d1, d2, df=False):
    """
    d1: point 1
    d2: point 2
    df: if points are Data Frames
    """
    if df:
        return d2.apply(lambda row: np.sqrt(np.sum((d1.to_numpy() - row.to_numpy()) ** 2)), axis=1)
    return np.sqrt(np.sum((d1 - d2) ** 2))

def manhattan_dist(d1, d2, df=False):
    """
    d1: point 1
    d2: point 2
    df: if points are Data Frames
    """
    if df:
        return d2.apply(lambda row: np.sum(abs(d1.to_numpy() - row.to_numpy())), axis=1)
    return np.sum(abs(d1 - d2))

def cosine_sim(d1, d2, df=False):
    """
    d1: point 1
    d2: point 2
    df: if points are Data Frames
    """
    if df:
        return d2.apply(lambda row: np.dot(d1.to_numpy(), row.to_numpy()) / (np.linalg.norm(d1.to_numpy()) * np.linalg.norm(row.to_numpy())), axis=1)
    return np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2))

def k_means_plusplus(D, k, dist):
    if dist == 1:
        distances = D.apply(euclidean_dist, args=(D, True), axis=1)
    elif dist == 2:
        distances = D.apply(manhattan_dist, args=(D, True), axis=1)
    else:
        distances = D.apply(cosine_sim, args=(D, True), axis=1)

    centroids = [0] * k
    if dist != 3:
        centroids[0], centroids[1] = np.unravel_index(np.argmax(distances), distances.shape)
        distances.loc[centroids[:2], centroids[:2]] = 0

        for i in range(2, k):
            # Iterate through all points x in D and find the point x s.t. the sum of distances from x to the previously selected centroids (m1, m2, ..., m_i-1) is maximum
            centroids[i] = np.argmax(distances[centroids[:i]].apply(lambda row: np.sum(row), axis=1))
            # So this points cannot be selected as a centroids again
            distances.loc[centroids[i], centroids[:i]] = 0
    else:
        centroids[0], centroids[1] = np.unravel_index(np.argmin(distances), distances.shape)
        distances.loc[centroids[:2], centroids[:2]] = np.Infinity
        for i in range(2, k):
            # Iterate through all points x in D and find the point x s.t. the sum of distances from x to the previously selected centroids (m1, m2, ..., m_i-1) is minimum
            centroids[i] = np.argmin(distances[centroids[:i]].apply(lambda row: np.sum(row), axis=1))
            distances.loc[centroids[i], centroids[:i]] = np.Infinity

    return centroids
    
def k_means(D, k, initial, dist, stand, epsilon):
    if stand:
        # Min-max standardization
        for col, val in D.items():
            min_val = val.min()
            max_val = val.max()
            D[col] = (val - min_val) / (max_val - min_val)

    cl = [None] * D.shape[0]
    if initial == 0:
        m = D.sample(k, replace=False).values
    else:
        m = [D.iloc[i].to_list() for i in k_means_plusplus(D, k, dist)]

    flag = True
    while flag:
        m_old = m.copy()
        cl = [Cluster(i, D.shape[1]) for i in m]

        for index, x in D.iterrows():
            if dist == 1:
                cluster = np.argmin([euclidean_dist(c.get_centroid(), np.array(x)) for c in cl])
            elif dist == 2:
                cluster = np.argmin([manhattan_dist(c.get_centroid(), np.array(x)) for c in cl])
            else:
                cluster = np.argmax([cosine_sim(c.get_centroid(), np.array(x)) for c in cl])

            cl[cluster].add(index, x.to_list())
        
        for j in range(k):
            m[j] = cl[j].mean()

        # Stoppage Conditions
        if np.array_equal(np.sort(m), np.sort(m_old)):
            flag = False
        
        if dist == 1:
            if np.max([euclidean_dist(m[j], m_old[j]) for j in range(k)]) <= epsilon:
                flag = False
        elif dist == 2:
            if np.max([manhattan_dist(m[j], m_old[j]) for j in range(k)]) <= epsilon:
                flag = False
        else:
            if np.min([cosine_sim(m[j], m_old[j]) for j in range(k)]) >= epsilon:
                flag = False

    return cl

def main(argv):
    D = pd.read_csv(argv[1], skiprows=[0], header=None, dtype=str)
    restriction = pd.read_csv(argv[1], nrows=1, header=None).iloc[0].to_list()
    K = int(argv[2])
    initial_cluster = int(argv[3])
    distance = int(argv[4])
    standardize = bool(int(argv[5]))
    epsilon = float(argv[6])
    name = argv[1].split("/")[-1] if "/" in argv[1] else argv[1].split("\\")[-1]
    D_filtered =  D.loc[:, map(bool, restriction)].copy()
    D_filtered.columns = range(D_filtered.shape[1])
    D_filtered = D_filtered.astype(float)

    clusters = k_means(D_filtered, K, initial_cluster, distance, standardize, epsilon)

    D['cluster'] = None
    with open(f".\\results_kmeans\\{name[:-4]}.out.txt", "w") as f:
        f.write(f"Output for python3 {' '.join(argv)}\n\n")
        f.write(f'Initial Centroid: {"Random" if distance == 0 else "K-means++"}\n')

        if distance == 1:
            f.write(f'Distance Metric: Euclidean Distance\n')
        elif distance == 2:
            f.write(f'Distance Metric: Manhattan Distance\n')
        else:
            f.write(f'Distance Metric: Cosine Simularity\n')

        f.write(f'Standardization: {"Min-Max Standardization" if standardize else "None"}\n')
        f.write(f'Stoppage Threshold: {epsilon}\n\n')

        f.write('-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-\n\n')
        for j in range(len(clusters)):
            centroid = clusters[j].get_centroid()

            if distance == 1:
                dists = [euclidean_dist(D_filtered.iloc[x], centroid) for x  in clusters[j].get_points()]
            elif distance == 2:
                dists = [manhattan_dist(D_filtered.iloc[x], centroid) for x  in clusters[j].get_points()]
            else:
                dists = [cosine_sim(D_filtered.iloc[x], centroid) for x  in clusters[j].get_points()]

            f.write(f'Cluster {j}:\nCenter: ')
            for i in clusters[j].get_centroid():
                f.write(f'{i},')
            f.write('\n')
            f.write(f'Max Dist. to Center: {np.max(dists)}\n')
            f.write(f'Min Dist. to Center: {np.min(dists)}\n')
            f.write(f'Avg Dist. to Center: {np.mean(dists)}\n')
            f.write(f'SSE for Cluster: {np.sum(np.array(dists)**2)}\n\n')
            f.write(f'{clusters[j].get_num()} Points:\n')

            for x in clusters[j].get_points():
                D['cluster'].iloc[x] = j
                for i in D.drop('cluster', axis=1).iloc[x].to_list():
                    f.write(f'{i},')
                f.write('\n')

            f.write('\n')
            f.write('-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-\n\n')

        D.to_csv(f".\\data_clustered\\kmeans\\{name[:-4]}_clustered.csv", index=False)

if __name__ == "__main__":
    main(argv)
