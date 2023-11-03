"""
Course: CSC 466
Quarter: Fall 2023
Assigment: Lab 4

Name(s):
    Sophia Chung // [ADD CALPOLY EMAIL]
    Andrew Kerr // adkerr@calpoly.edu

Description:
    How to run: python3 kmeans.py 
                    <input file: consits of data to build classifier> 
                    <K: number of neighbors to consider> 
                    <distance metric: 1 - eucledian, 2 - manhattan, 3 - cosine sim>
                    <min-max standardization: 1 - preform, 0 - do not preform>
"""
import pandas as pd
import numpy as np
from sys import argv


def eucledian_dist(d1, d2):
    """
    d1: row to predict
    d2: data to predict on
    """
    return d2.apply(lambda row: np.sqrt(np.sum((d1.to_numpy() - row.to_numpy()) ** 2)), axis=1)

def manhattan_dist(d1, d2):
    """
    d1: row to predict
    d2: data to predict on
    """
    return d2.apply(lambda row: np.sum(abs(d1.to_numpy() - row.to_numpy())), axis=1)

def cosine_sim(d1, d2):
    """
    d1: row to predict
    d2: data to predict on
    """
    return d2.apply(lambda row: np.dot(d1.to_numpy(), row.to_numpy()) / (np.linalg.norm(d1.to_numpy()) * np.linalg.norm(row.to_numpy())), axis=1)

def k_means(D, k, dist, stand):
    if stand:
        # Min-max standardization
        for col, val in D.items():
            min_val = val.min()
            max_val = val.max()
            D[col] = (val - min_val) / (max_val - min_val)

    if dist == 1:
        distances = D.apply(eucledian_dist, args=(D,), axis=1)
    elif dist == 2:
        distances = D.apply(manhattan_dist, args=(D,), axis=1)
    else:
        distances = D.apply(cosine_sim, args=(D,), axis=1)

    # if dist != 3:
    #     closest_neighbors = distances.T.apply(lambda row: row.nsmallest(k).index.tolist(), axis=1)
    # else:
    #     closest_neighbors = distances.T.apply(lambda row: row.nlargest(k).index.tolist(), axis=1)
    
    # predictions = {}
    # for i, row in enumerate(closest_neighbors):
    #     predictions[D.iloc[i]["index"]] = D.iloc[row][C].value_counts().index.tolist()[0]

    return predictions

def main(argv):
    D = pd.read_csv(argv[1], skiprows=[0], header=None, dtype=str)
    restriction = pd.read_csv(argv[1], nrows=1, header=None).iloc[0].to_list()
    K = int(argv[2])
    distance = int(argv[3])
    standardize = bool(argv[4])
    name = argv[1].split("/")[-1] if "/" in argv[1] else argv[1].split("\\")[-1]

    D =  D.loc[:, map(bool, restriction)]
    D.columns = range(D.shape[1])
    D = D.astype(float)

    k_means(D, K, distance, standardize)

    # predictions = knn_classifier_same(D_clean, A, C, K, dist) if flag else knn_classifier_different(D_clean, A, C, classify.copy(), K, dist)
    # D["pred_class"] = D.index.map(predictions)
    
    # D.to_csv(f".\\results_KNN\\{name[:-4]}-results.out.csv", index=False)

    # confusion = confusion_matrix(D, C, D[C].unique())
    # accuracy = np.sum(np.diag(confusion)) / np.sum(confusion.to_numpy())

    # if flag:
    #     with open(f".\\results_KNN\\{name[:-4]}-metrics.out.txt", "w") as f:
    #         f.write(f"Output for python3 {' '.join(argv)}\n\n")
    #         f.write(f"Num Neighbors: {K}\n")

    #         if dist == 1:
    #             f.write("Distance Metric: Eucledian Distance\n\n")
    #         elif dist == 2:
    #             f.write("Distance Metric: Manhattan Distance\n\n")
    #         else:
    #             f.write("Distance Metric: Cosine Similarity\n\n")

    #         f.write(f"Confusion Matrix:\n{confusion}\n\n")
    #         f.write(f"Accuracy: {round(accuracy, 4)}\n")

if __name__ == "__main__":
    main(argv)
