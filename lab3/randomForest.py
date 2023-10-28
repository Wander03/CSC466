"""
Course: CSC 466
Quarter: Fall 2023
Assigment: Lab 3

Name(s):
    Brendan Callender // bscallen@calpoly.edu
    Andrew Kerr // adkerr@calpoly.edu

Description:
    How to run: python3 randomForest.py 
                    <input file> 
                    <NumAttributes> 
                    <NumDataPoints> 
                    <NumTrees>
                    <number of folds; 0 = no cross-validation, -1 = all-but-one cross-validation> 
                    <threshold value> 
                    <1 if gain ratio, 0 if gain> 
"""
import pandas as pd
import numpy as np
from sys import argv
from random import sample
from InduceC45 import C45
from classify import confusion_matrix
from validation import predict_contain, compute_metrics


def random_forest_classifier(D, A, C, threshold, gain, m, k, N):
    """
    Inputs:
        D - Pandas DataFrame 
        A - Dict of Attributes, Type (if Type > 0 cat, else quant)
        C - String of DataFrame column with class variable
        threshold - Minimum accepted entropy
        ratio - 1 if gain ratio, 0 if gain
        m - NumAttributes: this parameter controls how many attributes each decision tree built by the Random
                           Forest classifier shall contain.
        k - NumDataPoints: the number of data points selected randomly with replacement to form a dataset for
                           each decision tree.
        N - NumTrees: the number of the decision trees to build.

    Outputs:
        Constructed random forest 
    """
    A_lst = list(A.items())
    forest = [None] * N

    for n in range(N):
        A_sub = dict(sample(A_lst, m))
        D_sub = D.sample(k, replace=True)
        forest[n] = C45(D_sub, A_sub, C, threshold, gain)

    return forest

def main(argv):
    D = pd.read_csv(argv[1], skiprows=[1, 2], dtype=str)
    A = D.columns.to_list()
    sizes = pd.read_csv(argv[1], skiprows=[0], nrows=1, header=None).iloc[0].to_list()
    C = pd.read_csv(argv[1], skiprows=[0, 1], nrows=1, header=None)[0][0]
    name = argv[1].split("/")[-1] if "/" in argv[1] else argv[1].split("\\")[-1]

    NumAttributes = int(argv[2])
    NumDataPoints = int(argv[3])
    NumTrees = int(argv[4])
    n_folds = int(argv[5])
    threshold = float(argv[6])
    gain = int(argv[7])

    A = dict(zip(A, sizes))
    del A[C]
    for k, v in A.copy().items():
        if v < 0:
            del A[k]

    votes = np.zeros((D.shape[0], len(D[C].unique())))
    class_to_vote = {value: number for number, value in enumerate(D[C].unique())}
    vote_to_class = {number: value for number, value in enumerate(D[C].unique())}

    # CV
    if n_folds > 1:
        # V-Fold creation
        D_random = D.sample(frac=1).copy()
        fold_size = len(D_random) // n_folds
        D_random["Fold"] = np.append(np.repeat(range(1, n_folds+1), fold_size), np.repeat(n_folds, len(D_random) % n_folds))

        for fold in range(1, n_folds+1):
            D_train = D_random[D_random["Fold"] != fold].copy()
            D_test = D_random[D_random["Fold"] == fold].copy()

            forest = random_forest_classifier(D_train, A, C, threshold, gain, NumAttributes, NumDataPoints, NumTrees)
            for tree in forest:
                D_test["pred_class"] = D_test.apply(predict_contain, args=(tree,C), axis=1)
                D_test["pred_class"] = D_test["pred_class"].map(class_to_vote)
                for i, row in D_test.iterrows():
                    votes[i, row["pred_class"]] += 1

    elif n_folds == -1: 
        # all-but-one CV
        for i in D.index:
            D_train = D[D.index != i].copy()
            D_test = D[D.index == i].copy()
            forest = random_forest_classifier(D_train, A, C, threshold, gain, NumAttributes, NumDataPoints, NumTrees)
            for tree in forest:
                D_test["pred_class"] = D_test.apply(predict_contain, args=(tree,C), axis=1)
                D_test["pred_class"] = D_test["pred_class"].map(class_to_vote)
                for i, row in D_test.iterrows():
                    votes[i, row["pred_class"]] += 1

    else:
        # No CV
        forest = random_forest_classifier(D, A, C, threshold, gain, NumAttributes, NumDataPoints, NumTrees)

        D["pred_class"] = D.apply(predict_contain, args=(forest[0],C), axis=1)
        D["pred_class"] = D["pred_class"].map(class_to_vote)
        for i, row in D.iterrows():
            votes[i, row["pred_class"]] += 1

    plurality_votes = np.vectorize(vote_to_class.get)(np.argmax(votes, axis=1))
    D["pred_class"] = plurality_votes

    D.to_csv(f".\\results_RF\\{name[:-4]}-results.out.csv", index=False)

    confusion = confusion_matrix(D, C, D[C].unique())
    accuracy = np.sum(np.diag(confusion)) / np.sum(confusion.to_numpy())

    with open(f".\\results_RF\\{name[:-4]}-metrics.out.txt", "w") as f:
        f.write(f"Output for python3 {' '.join(argv)}\n\n")
        f.write(f"NumAttributes: {NumAttributes}\nNumDataPoints: {NumDataPoints}\nNumTrees: {NumTrees}\n\n")
        f.write(f"Threshold: {threshold}\nUsing: {'Gain' if gain == 0 else 'Gain Ratio'}\nFolds: {n_folds if n_folds >= 0 else 'all-but-one'}\n\n")
        f.write(f"Confusion Matrix:\n{confusion}\n\n")
        f.write(f"Accuracy: {round(accuracy, 4)}\n")

if __name__ == "__main__":
    main(argv)
