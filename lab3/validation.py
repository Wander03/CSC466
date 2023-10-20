"""
Course: CSC 466
Quarter: Fall 2023
Assigment: Lab 3

Name(s):
    Brendan Callender // bscallen@calpoly.edu
    Andrew Kerr // adkerr@calpoly.edu

Description:
    How to run: <training data> <number of folds; 0 = no corss-validation, -1 = all-but-one cross-validation> <OPTIONAL: restictions file>
"""
import pandas as pd
import numpy as np
from InduceC45 import C45
from classify import predict_contain, confusion_matrix
from sys import argv

def K_Fold_cross_validation():
    pass

def main(argv):
    D = pd.read_csv(argv[1], skiprows=[1, 2], dtype=str)
    A = D.columns.to_list()
    sizes = pd.read_csv(argv[1], skiprows=[0], nrows=1, header=None).iloc[0].to_list()
    C = pd.read_csv(argv[1], skiprows=[0, 1], nrows=1, header=None)[0][0]
    name = argv[1].split("/")[-1]

    A.remove(C)
    for i, s in enumerate(sizes):
        if s <= 0:
            A.remove(A[i])

    try:
        restrict = pd.read_csv(argv[3], header=None).values.tolist()[0]
        A = [a for a, v in zip(A, restrict) if v == 1]
    except Exception as e:
        print("No Restriction File Provided (Using All Columns)")

    n_folds = int(argv[2])

    if n_folds > 0:
        # V-Fold CV
        D = D.sample(frac=1, random_state=43)
        fold_size = len(D) // n_folds
        D["Fold"] = np.append(np.repeat(range(1, n_folds+1), fold_size), np.repeat(n_folds, len(D) % n_folds))

        for fold in range(1, n_folds+1):
            D_train = D[D["Fold"] != fold].copy()
            D_test = D[D["Fold"] == fold].copy()
            T = C45(D_train, A, C, 0, 0)
            D_test["pred_class"] = D_test.apply(predict_contain, args=(T,C), axis=1)
            print(np.diag(confusion_matrix(D_test, C, D[C].unique())))
            print((confusion_matrix(D_test, C, D[C].unique())))

    elif n_folds == -1: 
        # all-but-one CV
        for i in D.index:
            D_train = D[D.index != i].copy()
            D_test = D[D.index == i].copy()
            T = C45(D_train, A, C, 0, 0)
            D_test["pred_class"] = D_test.apply(predict_contain, args=(T,C), axis=1)
            print(np.diag(confusion_matrix(D_test, C, D[C].unique())))
            print((confusion_matrix(D_test, C, D[C].unique())))
    else:
        # No CV
        T = C45(D, A, C, 0, 0)
        D["pred_class"] = D.apply(predict_contain, args=(T,C), axis=1)
        print(np.diag(confusion_matrix(D, C, D[C].unique())))
        print((confusion_matrix(D, C, D[C].unique())))

if __name__ == "__main__":
    main(argv)
