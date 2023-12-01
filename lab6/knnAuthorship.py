"""
Course: CSC 466
Quarter: Fall 2023
Assigment: Lab 6

Name(s):
    Brendan Callender // bscallen@calpoly.edu
    Andrew Kerr // adkerr@calpoly.edu

Description:
    How to run: python3 knn.py
                    <input file: consits of data to build classifier> 
                    <K: number of neighbors to consider> 
                    <distance metric: 0 - cosine, 1 - okapi>
"""
import pandas as pd
import numpy as np
from sys import argv
from Vector import Vector


def knn_classifier(D, C, k, dist):
    # REWORK FOr USE OF VECTORS AND MAKE SURE TO IGNORE ITSELF (REMOVE EXACTY THE SAME =1???)
    if dist != 3:
        closest_neighbors = distances.T.apply(lambda row: row.nsmallest(k).index.tolist(), axis=1)
    else:
        closest_neighbors = distances.T.apply(lambda row: row.nlargest(k).index.tolist(), axis=1)
    
    predictions = {}
    for i, row in enumerate(closest_neighbors):
        predictions[D.iloc[i]["index"]] = D.iloc[row][C].value_counts().index.tolist()[0]

    return predictions

def main(argv):
    ground_truth = pd.read_csv('./vectorized_data/ground_truth.csv')
    df = pd.read_csv('./vectorized_data/doc_freqs.csv')
    tf = pd.read_csv('./vectorized_data/term_freqs.csv')
    tf_idf = pd.read_csv('./vectorized_data/tf_idf.csv')
    f = Vector(ground_truth['file'][1], ground_truth['author'][1], tf.iloc[1], tf_idf.iloc[1], ground_truth['size'][1])
    g = Vector(ground_truth['file'][0], ground_truth['author'][0], tf.iloc[0], tf_idf.iloc[0], ground_truth['size'][0]) 
    avg = ground_truth['size'].mean()
    print(g.okapi_similarity(f, df, avg))
    breakpoint()
    # fix okapi? ordering of docs consistent?

    # # UPDATE THIS
    # D = pd.read_csv(argv[1], skiprows=[1, 2], dtype=str)
    # K = int(argv[3])
    # dist = int(argv[4])

    # predictions = knn_classifier_same(D_clean, A, C, K, dist) if flag else knn_classifier_different(D_clean, A, C, classify.copy(), K, dist)
    # D["pred_class"] = D.index.map(predictions)
    
    # D.to_csv(f".\\results_KNN\\{name[:-4]}-results.out.csv", index=False)

if __name__ == "__main__":
    main(argv)
