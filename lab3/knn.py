"""
Course: CSC 466
Quarter: Fall 2023
Assigment: Lab 3

Name(s):
    Brendan Callender // bscallen@calpoly.edu
    Andrew Kerr // adkerr@calpoly.edu

Description:
    How to run: python3 knn.py 
                    <input file: consits of data to build classifier> 
                    <classify file: consists of data to be classified>
                    <K: number of neighbors to consider> 
                    <distance metric: 1 - eucledian, 2 - manhattan, 3 - cosine sim>
                    <OPTIONAL: restrictions file>
"""
import pandas as pd
import numpy as np
from sys import argv
from classify import confusion_matrix


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

def knn_classifier_different(D, A, C, classify, k, dist):
    categorical_col = [key for key, value in A.items() if value != 0]

    D_dummy = pd.get_dummies(D[categorical_col], columns=categorical_col)
    D = pd.concat([D, D_dummy], axis=1)
    D.drop(categorical_col, axis=1, inplace=True)

    classify_dummy = pd.get_dummies(classify[categorical_col], columns=categorical_col)
    classify = pd.concat([classify, classify_dummy], axis=1)
    classify.drop(categorical_col, axis=1, inplace=True)

    # Min-max standardization
    for col, val in A.items():
        if val == 0:
            D[col] = D[col].astype(float)
            min_val = D[col].min()
            max_val = D[col].max()
            D[col] = (D[col] - min_val) / (max_val - min_val)

            classify[col] = classify[col].astype(float)
            min_val = classify[col].min()
            max_val = classify[col].max()
            classify[col] = (classify[col] - min_val) / (max_val - min_val)

    if dist == 1:
        distances = D.drop(C, axis=1).apply(eucledian_dist, args=(classify,), axis=1)
    elif dist == 2:
        distances = D.drop(C, axis=1).apply(manhattan_dist, args=(classify,), axis=1)
    else:
        distances = D.drop(C, axis=1).apply(cosine_sim, args=(classify,), axis=1)

    if dist != 3:
        closest_neighbors = distances.T.apply(lambda row: row.nsmallest(k).index.tolist(), axis=1)
    else:
        closest_neighbors = distances.T.apply(lambda row: row.nlargest(k).index.tolist(), axis=1)
    
    predictions = {}
    for i, row in enumerate(closest_neighbors):
        predictions[D.iloc[i]["index"]] = D.iloc[row][C].value_counts().index.tolist()[0]

    return predictions
    
def knn_classifier_same(D, A, C, k, dist):
    categorical_col = [key for key, value in A.items() if value != 0]
    D_dummy = pd.get_dummies(D[categorical_col], columns=categorical_col)
    D = pd.concat([D, D_dummy], axis=1)
    D.drop(categorical_col, axis=1, inplace=True)

    # Min-max standardization
    for col, val in A.items():
        if val == 0:
            D[col] = D[col].astype(float)
            min_val = D[col].min()
            max_val = D[col].max()
            D[col] = (D[col] - min_val) / (max_val - min_val)


    if dist == 1:
        distances = D.drop(C, axis=1).apply(eucledian_dist, args=(D.drop(C, axis=1),), axis=1)
    elif dist == 2:
        distances = D.drop(C, axis=1).apply(manhattan_dist, args=(D.drop(C, axis=1),), axis=1)
    else:
        distances = D.drop(C, axis=1).apply(cosine_sim, args=(D.drop(C, axis=1),), axis=1)

    if dist != 3:
        closest_neighbors = distances.T.apply(lambda row: row.nsmallest(k).index.tolist(), axis=1)
    else:
        closest_neighbors = distances.T.apply(lambda row: row.nlargest(k).index.tolist(), axis=1)
    
    predictions = {}
    for i, row in enumerate(closest_neighbors):
        predictions[D.iloc[i]["index"]] = D.iloc[row][C].value_counts().index.tolist()[0]

    return predictions

def main(argv):
    D = pd.read_csv(argv[1], skiprows=[1, 2], dtype=str)
    A = D.columns.to_list()
    sizes = pd.read_csv(argv[1], skiprows=[0], nrows=1, header=None).iloc[0].to_list()
    C = pd.read_csv(argv[1], skiprows=[0, 1], nrows=1, header=None)[0][0]
    name = argv[1].split("/")[-1] if "/" in argv[1] else argv[1].split("\\")[-1]

    D_clean = D.drop(D[D.apply(lambda row: any(row == '?'), axis=1)].index).reset_index().copy()

    flag = True
    if argv[1] != argv[2]:
        classify = pd.read_csv(argv[2], skiprows=[1, 2], dtype=str)
        flag = False

    K = int(argv[3])
    dist = int(argv[4])

    try:
        restrict = pd.read_csv(argv[5], header=None).values.tolist()[0]
        for a, v in zip(A.copy(), restrict):
            if v != 1:
                A.remove(a)
    except Exception as e:
        print("No Restriction File Provided (Using All Columns)")

    A = dict(zip(A, sizes))
    del A[C]
    for k, v in A.copy().items():
        if v < 0:
            del A[k]

    predictions = knn_classifier_same(D_clean, A, C, K, dist) if flag else knn_classifier_different(D_clean, A, C, classify.copy(), K, dist)
    D["pred_class"] = D.index.map(predictions)
    
    D.to_csv(f".\\results_KNN\\{name[:-4]}-results.out.csv", index=False)

    confusion = confusion_matrix(D, C, D[C].unique())
    accuracy = np.sum(np.diag(confusion)) / np.sum(confusion.to_numpy())

    with open(f".\\results_KNN\\{name[:-4]}-metrics.out.txt", "w") as f:
        f.write(f"Output for python3 {' '.join(argv)}\n\n")
        f.write(f"Num Neighbors: {K}\n")

        if dist == 1:
            f.write("Distance Metric: Eucledian Distance\n\n")
        elif dist == 2:
            f.write("Distance Metric: Manhattan Distance\n\n")
        else:
            f.write("Distance Metric: Cosine Similarity\n\n")

        f.write(f"Confusion Matrix:\n{confusion}\n\n")
        f.write(f"Accuracy: {round(accuracy, 4)}\n")

if __name__ == "__main__":
    main(argv)
