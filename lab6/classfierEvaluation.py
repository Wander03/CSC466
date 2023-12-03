"""
Course: CSC 466
Quarter: Fall 2023
Assigment: Lab 6

Name(s):
    Brendan Callender // bscallen@calpoly.edu
    Andrew Kerr // adkerr@calpoly.edu

Description:

"""
import pandas as pd
import numpy as np
from sys import argv


def confusion_matrix(D, C, c):
    D[C] = pd.Categorical(D[C], categories=c)
    D['pred_author'] = pd.Categorical(D['pred_author'], categories=c)
    return(pd.crosstab(D[C], D['pred_author']).reindex(index=c, columns=c, fill_value=0))

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

    return [precision.fillna(0), recall.fillna(0), overall_accuracy, overall_error_rate, average_accuracy, average_error_rate]

def main(argv):
    ground_truth = pd.read_csv(argv[1]).drop('size', axis=1)
    predictions = pd.read_csv(argv[2])
    knn = bool(argv[3])
    combo = ground_truth.merge(predictions, how='left', on='file')

    for a in combo['author'].unique():
        a_bool = combo[combo['author'] == a]['pred_author'] == a
        a_boolb = combo[combo['author'] != a]['pred_author'] == a
        TP = a_bool.sum()
        FP = a_boolb.sum()
        FN = a_bool.shape[0] - TP
        print('-' * 80)
        print('Author:', a)
        print('Hits (correctly predicted):', TP)
        print('Strikes (false positives):', FP)
        print('Misses (false negatives):', FN)
        if knn:
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            f1 = 2 * (precision * recall) / (precision + recall)
            print('Precision:', precision)
            print('Recall:', recall)
            print('F1-Score:', f1)
        print('-' * 80)

    full_confusion = confusion_matrix(combo, 'author', combo['author'].unique())
    full_accuracy = np.sum(np.diag(full_confusion)) / np.sum(full_confusion.to_numpy())

    print('Overall Accuracy:', full_accuracy)
    print('-' * 80)

    full_confusion.to_csv(f".\\confusion\\confusion_matrix.csv", index=False)

if __name__ == "__main__":
    main(argv)
