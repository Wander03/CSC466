"""
Course: CSC 466
Quarter: Fall 2023
Assigment: Lab 3

Name(s):
    Brendan Callender // bscallen@calpoly.edu
    Andrew Kerr // adkerr@calpoly.edu

Description:
    How to run: python3 InduceC45 <input file> <threshold value> <1 if gain ratio, 0 if gain> <OPTIONAL: restrictions file>
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
    p_lst = np.where(p_lst == 0, 1, p_lst)
    return -1 * np.sum(p_lst * np.log2(p_lst))

def gain(D, C, a):
    return entropy(D[C]) - entropySub(D, C, a)

def findBestSplit(D, C, a, ratio=False):
    p0 = entropy(D[C])
    props = D.groupby(a)[C].value_counts().sort_index()
    alphas = props.index.get_level_values(0).unique()
    pk = np.cumsum(props.unstack().fillna(0).to_numpy(), axis=0)
    n = D.shape[0]

    if pk.shape[0] == 1:
        return (0, 0) if not ratio else (0, 0, 0)

    gain_lst = []
    for r in range(0, pk.shape[0]-1):
        p_lst_lower = pk[r] / n
        p_lst_lower = np.where(p_lst_lower == 0, 1, p_lst_lower)
        entropy_lower = np.sum(pk[r]) / n * (-1*np.sum(p_lst_lower * np.log2(p_lst_lower)))
        
        p_lst_upper = (pk[-1]-pk[r]) / n
        p_lst_upper = np.where(p_lst_upper == 0, 1, p_lst_upper)
        entropy_upper = (n-np.sum(pk[r])) / n * (-1*np.sum(p_lst_upper * np.log2(p_lst_upper)))

        gain_lst.append(p0 - (entropy_lower + entropy_upper))

    best = gain_lst.index(max(gain_lst))
    return (alphas[best], gain_lst[best]) if not ratio else (alphas[best], gain_lst[best], entropy_lower + entropy_upper)

def selectSplittingAttribute(D, A, C, threshold, ratio=False):
    G = dict()
    if ratio:
        for a in A.keys():
            if A[a] == 0:
                alpha, gain_val, entropy_val = findBestSplit(D, C, a, True)
                G[a] = (gain_val / entropy_val, alpha) if entropy_val != 0 else (0, alpha)
            else:
                G[a] = (gain(D, C, a) / entropy(D[a]), None)
    else:
        for a in A.keys():
            if A[a] == 0:
                alpha, gain_val = findBestSplit(D, C, a)
                G[a] = (gain_val, alpha)
            else:
                G[a] = (gain(D, C, a), None)

    best = max(G, key=lambda k: G[k][0])
    return (best, G[best][1]) if G[best][0] > threshold else (None, None)
        

def C45(D, A, C, threshold, ratio):
    """
    Inputs:
        D - Pandas DataFrame of training data
        A - Dict of Attributes, Type (if Type > 0 cat, else quant)
        C - String of DataFrame column with class variable
        threshold - Minimum accepted entropy

    Outputs:
        Constructed decision tree 
    """
    if len(D[C].unique()) == 1:
        T = {"leaf": {"decision": D[C][0], "p": 1}}
    elif not A.keys():
        T = {"leaf": {"decision": D[C].value_counts().index[0], "p": D[C].value_counts(normalize=True).values[0]}}
    else:
        a, val = selectSplittingAttribute(D, A, C, threshold, ratio)

        if a is None:
            T = {"leaf": {"decision": D[C].value_counts().index[0], "p": D[C].value_counts(normalize=True).values[0]}}
        elif val is None:
            plurality = D[C].value_counts(normalize=True)
            T = {"node": {"var": a, "type": 1, "plurality": plurality.index.tolist()[0], "p": plurality.values[0], "edges": []}}
            for v in D[a].unique():
                D_v = D[D[a] == v].reset_index(drop=True)
                A_v = A.copy()
                del A_v[a]
                edge = {"edge": {"value": v, **C45(D_v, A_v, C, threshold, ratio)}}
                T["node"]["edges"].append(edge)
        else:
            plurality = D[C].value_counts(normalize=True)
            T = {"node": {"var": a, "type": 0, "plurality": plurality.index.tolist()[0], "p": plurality.values[0], "edges": []}}
            edge1 = {"edge": {"alpha": val, "direction": "<=", **C45(D[D[a] <= val].reset_index(drop=True), A, C, threshold, ratio)}}
            edge2 = {"edge": {"alpha": val, "direction": ">", **C45(D[D[a] > val].reset_index(drop=True), A, C, threshold, ratio)}}
            T["node"]["edges"].append(edge1)
            T["node"]["edges"].append(edge2)

    return T
            
def main(argv):
    D = pd.read_csv(argv[1], skiprows=[1, 2], dtype=str)
    A = D.columns.to_list()
    sizes = pd.read_csv(argv[1], skiprows=[0], nrows=1, header=None).iloc[0].to_list()
    C = pd.read_csv(argv[1], skiprows=[0, 1], nrows=1, header=None)[0][0]
    name = argv[1].split("/")[-1] if "/" in argv[1] else argv[1].split("\\")[-1]

    try:
        restrict = pd.read_csv(argv[4], header=None).values.tolist()[0]
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
        elif v == 0:
            D[k] = D[k].astype(float)

    tree = {"dataset": name, **C45(D, A, C, float(argv[2]), int(argv[3]))}

    with open(f".\\trees\\{name[:-4]}Tree.json", "w") as f:
        f.write(json.dumps(tree, indent=4))

if __name__ == "__main__":
    main(argv)
