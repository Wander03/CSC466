import pandas as pd
import numpy as np
import json
from sys import argv


def predict():
    pass

def main(arvg):
    D = pd.read_csv(argv[1], skiprows=[1, 2], dtype=str)
    A = D.columns.to_list()
    C = pd.read_csv(argv[1], skiprows=[0, 1], nrows=1, header=None)[0][0]
    # T = pd.read_json(argv[2])
    T = pd.read_json(argv[2])

    print(T)

    # name = argv[1].split("/")[-1]

    # if C in A:
    #     A.remove(C)
    #     predict()
    # else:
    #     predict()
    

if __name__ == "__main__":
    main(argv)
