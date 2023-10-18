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
            else:
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
            else:
                pred = curr_node['plurality']
                if data[C] != pred:
                    classified_incorrect += 1
                classified_total += 1
                return pred

def main(argv):
    global classified_total
    global classified_incorrect

    D = pd.read_csv(argv[1], skiprows=[1, 2], dtype=str)
    A = D.columns.to_list()
    sizes = pd.read_csv(argv[1], skiprows=[0], nrows=1, header=None).iloc[0].to_list()
    C = pd.read_csv(argv[1], skiprows=[0, 1], nrows=1, header=None)[0][0]
    T = pd.read_json(argv[2])

    for i, s in enumerate(sizes):
        if s <= 0:
            A.remove(A[i])

    if C in A:
        D['pred_class'] = D.apply(predict_contain, args=(T,C), axis=1)
    else:
        D['pred_class'] = D.apply(predict_not_contain, args=(T,), axis=1)

    print(D)
    print(classified_total)
    print(classified_incorrect)

    print(f"Overall Accuracy: {(classified_total - classified_incorrect) / classified_total}")
    # print(f"Error Rate: {(classified_total - classified_incorrect) / }")

if __name__ == "__main__":
    main(argv)


# QUESTION: how do we know if the input data contains the class variable? --> depending how, how do we know the size of the predicted class? 