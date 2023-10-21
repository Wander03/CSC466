"""
Course: CSC 466
Quarter: Fall 2023
Assigment: Lab 3

Name(s):
    Brendan Callender // bscallen@calpoly.edu
    Andrew Kerr // adkerr@calpoly.edu

Description:
    How to run: 
    ptyhon3
        <training data> 
        <number of folds; 0 = no cross-validation, -1 = all-but-one cross-validation> 
        <threshold value> 
        <1 if gain ratio, 0 if gain> 
        <OPTIONAL: restictions file>
"""
import pandas as pd
import numpy as np
from InduceC45 import C45
from classify import predict_contain, confusion_matrix
from sys import argv


def compute_metrics(confusion, accuracy):
    TP = np.diag(confusion)
    FP = np.sum(confusion, axis=0) - TP
    FN = np.sum(confusion, axis=1) - TP

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    overall_accuracy = np.sum(TP) / np.sum(confusion.to_numpy())
    average_accuracy = np.mean(accuracy)
    overall_error_rate =  1 - overall_accuracy
    average_error_rate = 1 - average_accuracy

    # print(confusion)
    # print('\nPrecision:\n', precision.fillna(0))
    # print('\nRecall:\n', recall.fillna(0))
    # print('\nOverall Accuracy:\n', overall_accuracy)
    # print('\nOverall Error Rate:\n', overall_error_rate)
    # print('\nAverage Accuracy:\n', average_accuracy)
    # print('\nAverage Error Rate:\n', average_error_rate)

    return [precision.fillna(0), recall.fillna(0), overall_accuracy, overall_error_rate, average_accuracy, average_error_rate]

def main(argv):
    D = pd.read_csv(argv[1], skiprows=[1, 2], dtype=str)
    A = D.columns.to_list()
    sizes = pd.read_csv(argv[1], skiprows=[0], nrows=1, header=None).iloc[0].to_list()
    C = pd.read_csv(argv[1], skiprows=[0, 1], nrows=1, header=None)[0][0]
    name = argv[1].split("/")[-1] if "/" in argv[1] else argv[1].split("\\")[-1]
    threshold = float(argv[3])
    gain = int(argv[4])

    A.remove(C)
    for i, s in enumerate(sizes):
        if s <= 0:
            A.remove(A[i])

    try:
        restrict = pd.read_csv(argv[5], header=None).values.tolist()[0]
        A = [a for a, v in zip(A, restrict) if v == 1]
    except Exception as e:
        print("No Restriction File Provided (Using All Columns)")

    n_folds = int(argv[2])

    confusion = pd.DataFrame()
    accuracy = []
    error = []
    if n_folds > 0:
        # V-Fold CV
        D = D.sample(frac=1)
        fold_size = len(D) // n_folds
        D["Fold"] = np.append(np.repeat(range(1, n_folds+1), fold_size), np.repeat(n_folds, len(D) % n_folds))

        for fold in range(1, n_folds+1):
            D_train = D[D["Fold"] != fold].copy()
            D_test = D[D["Fold"] == fold].copy()
            T = C45(D_train, A, C, threshold, gain)
            D_test["pred_class"] = D_test.apply(predict_contain, args=(T,C), axis=1)
            confusion = confusion.add(confusion_matrix(D_test, C, D[C].unique()), fill_value=0)
            accuracy.append(np.sum(np.diag(confusion)) / np.sum(confusion.to_numpy()))

    elif n_folds == -1: 
        # all-but-one CV
        for i in D.index:
            D_train = D[D.index != i].copy()
            D_test = D[D.index == i].copy()
            T = C45(D_train, A, C, threshold, gain)
            D_test["pred_class"] = D_test.apply(predict_contain, args=(T,C), axis=1)
            confusion = confusion.add(confusion_matrix(D_test, C, D[C].unique()), fill_value=0)
            accuracy.append(np.sum(np.diag(confusion)) / np.sum(confusion.to_numpy()))

    else:
        # No CV
        T = C45(D, A, C, threshold, gain)
        D["pred_class"] = D.apply(predict_contain, args=(T,C), axis=1)
        print(np.diag(confusion_matrix(D, C, D[C].unique())))
        confusion = confusion_matrix(D, C, D[C].unique())
        accuracy.append(np.sum(np.diag(confusion)) / np.sum(confusion.to_numpy()))

    metrics = compute_metrics(confusion, accuracy)

    with open(f".\\results\\{name[:-4]}-results.out.csv", "w") as f:
        f.write(f"Output for python3 {' '.join(argv)}\n\n")
        f.write(f"Threshold: {threshold}\nUsing: {'Gain' if gain == 0 else 'Gain Ratio'}\nFolds: {n_folds}\n\n")
        f.write(f"Overall Confusion Matrix:\n{confusion}\n\n")
        f.write(f"Precision:\n{metrics[0]}\n\n")
        f.write(f"Recall:\n{metrics[1]}\n\n\n")
        f.write(f"Overall Accuracy: {round(metrics[2], 4)}\n")
        f.write(f"Overall Error Rate: {round(metrics[3], 4)}\n\n")
        f.write(f"Average Accuracy: {round(metrics[4], 4)}\n")
        f.write(f"Average Error Rate: {round(metrics[5], 4)}\n")

if __name__ == "__main__":
    main(argv)
