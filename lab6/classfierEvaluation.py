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

def main(argv):
    ground_truth = pd.read_csv(argv[1]).drop('size', axis=1)
    predictions = pd.read_csv(argv[2])
    knn = bool(int(argv[3]))
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
            precision = TP / (TP + FP) if TP + FP != 0 else 0
            recall = TP / (TP + FN)
            f1 = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0
            print('Precision:', precision)
            print('Recall:', recall)
            print('F1-Score:', f1)
        print('-' * 80)

    full_confusion = confusion_matrix(combo, 'author', combo['author'].unique())
    full_accuracy = np.sum(np.diag(full_confusion)) / np.sum(full_confusion.to_numpy())

    print('Overall Accuracy:', full_accuracy)
    print('-' * 80)

    full_confusion.insert(0, 'actual author', full_confusion.columns)

    full_confusion.to_csv(f".\\confusion\\confusion_matrix.csv", index=False)

if __name__ == "__main__":
    main(argv)
