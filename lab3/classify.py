"""
Course: CSC 466
Quarter: Fall 2023
Assigment: Lab 3

Name(s):
    Brendan Callender // bscallen@calpoly.edu
    Andrew Kerr // adkerr@calpoly.edu

Description:
    How to run: python3 classify.py <input file> <decision tree file> <OPTIONAL: 1 if silent run, 0 if not>
"""
import pandas as pd
from sys import argv


classified_total = 0
classified_incorrect = 0

def predict_not_contain(data, json):
    if 'leaf' in json.keys():
        return json['leaf']['decision']

    curr_node = json['node']
    while True:
        data_val = data[curr_node['var']]
        next_nodes = curr_node['edges']
        for e in next_nodes:
            if data_val == e['edge']['value']:
                if 'leaf' in e['edge'].keys():
                    return e['edge']['leaf']['decision']
                else:
                    curr_node = e['edge']['node']

        return curr_node['plurality']
            
def predict_contain(data, json, C):
    global classified_total
    global classified_incorrect

    if 'leaf' in json.keys():
        pred = json['leaf']['decision']
        if data[C] != pred:
            classified_incorrect += 1
        classified_total += 1
        return pred

    curr_node = json['node']
    while True:
        data_val = data[curr_node['var']]
        next_nodes = curr_node['edges']

        for e in next_nodes:
            if data_val == e['edge']['value']:
                if 'leaf' in e['edge'].keys():
                    pred = e['edge']['leaf']['decision']
                    if data[C] != pred:
                        classified_incorrect += 1
                    classified_total += 1
                    return pred
                else:
                    curr_node = e['edge']['node']

        pred = curr_node['plurality']
        if data[C] != pred:
            classified_incorrect += 1
        classified_total += 1
        return pred

def confusion_matrix(D, C, c):
    D[C] = pd.Categorical(D[C], categories=c)
    D['pred_class'] = pd.Categorical(D['pred_class'], categories=c)
    return(pd.crosstab(D[C], D['pred_class']).reindex(index=c, columns=c, fill_value=0))

def main(argv):
    global classified_total
    global classified_incorrect

    D = pd.read_csv(argv[1], skiprows=[1, 2], dtype=str)
    A = D.columns.to_list()
    sizes = pd.read_csv(argv[1], skiprows=[0], nrows=1, header=None).iloc[0].to_list()
    C = pd.read_csv(argv[1], skiprows=[0, 1], nrows=1, header=None)[0][0]
    T = pd.read_json(argv[2])
    
    try:
        silent = int(argv[3])
    except:
        silent = False

    A = dict(zip(A, sizes))
    del A[C]
    for k, v in A.copy().items():
        if v < 0:
            del A[k]

    if silent:
        D['pred_class'] = D.apply(predict_not_contain, args=(T,), axis=1)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        for i, p in zip(D.index, D.pred_class):
            print(f"{i}, {p}")
    else:
        if C in A.keys():
            D['pred_class'] = D.apply(predict_contain, args=(T,C), axis=1)
            print(f"Overall Accuracy: {round((classified_total - classified_incorrect) / classified_total, 4)}")
            print(f"Overall Error Rate: {round(classified_incorrect / classified_total, 4)}")
            print("\nConfusion Matrix:")
            print(confusion_matrix(D, C, D[C].unique()))
        else:
            D['pred_class'] = D.apply(predict_not_contain, args=(T,), axis=1)
            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_columns', None)
            print(D)

if __name__ == "__main__":
    main(argv)
