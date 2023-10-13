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
import json
from sys import argv


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
        

def C45(D, A, C, threshold, ratio):
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
        T = {"leaf": {"decision": D[C][0], "p": 1}}
    elif not A:
        T = {"leaf": {"decision": D[C].value_counts().index[0], "p": D[C].value_counts(normalize=True).values[0]}}
    else:
        a = selectSplittingAttribute(D, A, C, threshold, ratio)

        if a is None:
            T = {"leaf": {"decision": D[C].value_counts().index[0], "p": D[C].value_counts(normalize=True).values[0]}}
        else:
            T = {"node": {"var": a, "edges": []}}
            for v in D[a].unique():
                D_v = D[D[a] == v].reset_index(drop=True)
                edge = {"edge": {"value": v, **C45(D_v, list(set(A) - {a}), C, threshold, ratio)}}
                T["node"]["edges"].append(edge)
                
    return T
            
def main(argv):
    # tree = C45(pd.read_csv(".\\data\\nursery.csv", skiprows=[1, 2]), ["parents","has_nurs","form","children","housing","finance","social","health"], "class", .8, False)
    # tree = C45(pd.read_csv(".\\data\\adult-stretch.csv", skiprows=[1, 2]), ["Color","Size","Act","Age"], "Inflated", 0, False)

    D = pd.read_csv(argv[1], skiprows=[1, 2])
    A = D.columns.to_list()
    C = pd.read_csv(argv[1], skiprows=[0, 1], nrows=1, header=None)[0][0]
    A.remove(C)
    name = argv[1].split("/")[-1]

    try:
        restrict = pd.read_csv(argv[2], header=None, names=["data"])
        # restrict = restrict["data"].tolist()
        A = [a for a, v in zip(A, restrict) if v == "1"]
    except Exception as e:
        print(e)
    
    print(A)
    print(restrict["data"])

    # tree = {"dataset": name, **C45(D, A, C, 0, False)}

    # with open(f".\\out\\{name[:-4]}Tree.json", "w") as f:
    #     f.write(json.dumps(tree, indent=4))

if __name__ == "__main__":
    main(argv)
