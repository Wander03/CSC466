"""
Course: CSC 466
Quarter: Fall 2023
Assigment: Lab 3

Name(s):
    Brendan Callender // bscallen@calpoly.edu
    Andrew Kerr // adkerr@calpoly.edu

Description:
"""
import pandas as pd
import numpy as np


def entropySub(D, C, a):
    e_lst = D.groupby(a)[C].apply(entropy).to_numpy()
    p_lst = D[a].value_counts(normalize=True).values
    return np.sum(p_lst * e_lst)

def entropy(D):
    p_lst = D.value_counts(normalize=True).values
    return -1 * np.sum(p_lst * np.log2(p_lst))

def gain(D, C, a):
    return entropy(D[C]) - entropySub(D, C, a)

def selectSplittingAttribute(D, A, C, threshold, ratio=False):
    if ratio:
        G = [gain(D, C, a) / entropy(D[a]) for a in A]
    else:
        G = [gain(D, C, a) for a in A]

    best = G.index(max(G))
    return A[best] if G[best] > threshold else None
        

def C45(D, A, C, threshold):
    """
    Inputs:
        D - Pandas DataFrame of training data
        A - List of Attributes
        C - String of DataFrame column with class variable
        threshold - Minimum accepted entropy

    Outputs:
        Constructed decision tree 
    """
    if len(D[C].unique()) == 1:
        r = {
            "leaf": {
                "decision": D[C][0],
                "p": 1
                }
            }
        T = r
    elif not A:
        r = {
            "leaf": {
                "decision": D[C].value_counts().index[0],
                "p": 1
                }
            }
        T = r
    else:
        a = selectSplittingAttribute(D, A, C, threshold, True)

        if a is None:
            r = {
                "leaf": {
                    "decision": D[C].value_counts().index[0],
                    "p": 1
                    }
                }
            T = r
        else:
            r = {"value": a, "edge": []}
            T = r
            for v in D[a].unique():
                D_v = D[D[a] == v].reset_index()
                edge = C45(D_v, list(set(A) - {a}), C, threshold)
                r["edge"] = r["edge"].append(edge)
                
    return T
            

def main():
    C45(pd.read_csv(".\\data\\test.csv", skiprows=[1, 2]), ["Color","Size","Act","Age"], "Inflated", .5)

if __name__ == "__main__":
    main()
