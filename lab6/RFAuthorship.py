"""
Course: CSC 466
Quarter: Fall 2023
Assigment: Lab 6

Name(s):
    Brendan Callender // bscallen@calpoly.edu
    Andrew Kerr // adkerr@calpoly.edu

Description:
    How to run: python3 RFAuthorship.py
                    <ground truth file>
                    <input file: consits of data to build classifier> 
                    <Num Attributes (per tree)> 
                    <Num Data Points (per tree)>
                    <Num Trees (in forest)>
                    <threshold value> 
                    <1 if gain ratio, 0 if gain> 
                    <outfile name>
"""
import pandas as pd
import numpy as np
from sys import argv
from random import sample
from InduceC45 import C45
import datetime


def predict(data, json):
    if 'leaf' in json.keys():
        pred = json['leaf']['decision']
        return pred

    curr_node = json['node']
    while True:
        data_val = data[curr_node['var']]
        next_nodes = curr_node['edges']

        flag = True
        if curr_node["type"]:
            for e in next_nodes:
                if data_val == e['edge']['value']:
                    if 'leaf' in e['edge'].keys():
                        pred = e['edge']['leaf']['decision']
                    else:
                        curr_node = e['edge']['node']
                        flag = False
        else:
            if data_val == "?":
                pred = curr_node['plurality']
                return pred
            for e in next_nodes:
                if eval(f"{float(data_val)} {e['edge']['direction']} {float(e['edge']['alpha'])}"):
                    if 'leaf' in e['edge'].keys():
                        pred = e['edge']['leaf']['decision']
                        return pred
                    else:
                        curr_node = e['edge']['node']
                        flag = False
        if flag:
            pred = curr_node['plurality']
            return pred

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
    ground_truth = pd.read_csv(argv[1])
    tf_idf = pd.read_csv(argv[2])
    D = ground_truth.drop(['file', 'size'], axis=1).merge(tf_idf, how='left', left_index=True, right_index=True, suffixes=('_real', ''))

    A = D.columns.to_list()
    C = 'author_real'

    NumAttributes = int(argv[3])
    NumDataPoints = int(argv[4])
    NumTrees = int(argv[5])
    threshold = float(argv[6])
    gain = int(argv[7])
    outname = argv[8]

    A = dict(zip(A, [0] * len(A)))
    del A[C]

    votes = np.zeros((D.shape[0], len(D[C].unique())))
    class_to_vote = {value: number for number, value in enumerate(D[C].unique())}
    vote_to_class = {number: value for number, value in enumerate(D[C].unique())}

    # No CV
    forest = random_forest_classifier(D, A, C, threshold, gain, NumAttributes, NumDataPoints, NumTrees)

    for tree in forest:
        ground_truth["pred_author"] = D.apply(predict, args=(tree,), axis=1)
        ground_truth["pred_author"] = ground_truth["pred_author"].map(class_to_vote)
        for i, row in ground_truth.iterrows():
            votes[i, row["pred_author"]] += 1
    
    plurality_votes = np.vectorize(vote_to_class.get)(np.argmax(votes, axis=1))
    ground_truth["pred_author"] = plurality_votes

    ground_truth.drop(['author', 'size'], axis=1).to_csv(f".\\results_RF\\{outname}.csv", index=False)

if __name__ == "__main__":
    print(datetime.datetime.now())
    main(argv)
    print(datetime.datetime.now())
