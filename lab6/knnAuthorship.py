"""
Course: CSC 466
Quarter: Fall 2023
Assigment: Lab 6

Name(s):
    Brendan Callender // bscallen@calpoly.edu
    Andrew Kerr // adkerr@calpoly.edu

Description:
    How to run: python3 knnAuthorship.py
                    <input file: consits of data to build classifier> 
                    <K: number of neighbors to consider> 
                    <distance metric: 0 - cosine, 1 - okapi>
"""
import sys
import time
import pandas as pd
import numpy as np
import random
from sys import argv
from Vector import Vector


def cosine_sim(d1, d2):
    return d2.apply(lambda row: np.dot(d1.to_numpy(), row.to_numpy()) / (np.linalg.norm(d1.to_numpy()) * np.linalg.norm(row.to_numpy())), axis=1)


def knn_classifier(ground_truth, k, distances):
    closest_neighbors = distances.T.apply(lambda row: row.nlargest(k+1).index.tolist()[1:], axis=1)

    predictions = {}
    for i, row in enumerate(closest_neighbors):
        predictions[i] = ground_truth.iloc[row]['author'].value_counts().index.tolist()[0]

    return predictions

def main():

    start = time.time()
    args = sys.argv

    gt = args[1]
    ground_truth = pd.read_csv(gt)

    tfidf = args[2]
    tf_idf = pd.read_csv(tfidf)

    sim = int(args[3])

    if sim == 0:
        okapi=False
    elif sim == 1:
        okapi=True
    else:
        print('Please use valid similarity metric:')
        print('0: Cosine Similarity')
        print('1: Okapi')
        return None

    if okapi:
        k = int(args[4])
        df = args[5]
        tf = args[6]
        docf = pd.read_csv(df, na_filter=False)
        termf = pd.read_csv(tf)
        N = int(args[7])

        sample = random.sample(range(5000), N)

        vectors = []
        for i  in range(ground_truth.shape[0]):
            vectors.append(Vector(ground_truth['file'][i], ground_truth['author'][i], termf.iloc[i], tf_idf.iloc[i],
                   ground_truth['size'][i]))

        avg = ground_truth['size'].mean()
        distances = pd.DataFrame()
        for i in sample:
            v1 = vectors[i]
            sims = []
            for j in range(ground_truth.shape[0]):
                sims.append(v1.okapi_similarity(vectors[j], docf, avg))
            distances = pd.concat([distances, pd.Series(sims, name=i)], axis=1)

        predictions = knn_classifier(ground_truth, k, distances)
        temp = ground_truth.iloc[sample][['author', 'file']].reset_index(drop=True)
        preds = pd.Series(list(predictions.values()))
        temp['prediction'] = preds
        temp[['file', 'prediction']].to_csv('out/predictions.csv', index=False)


    else:
        k = int(args[4])
        distances = pd.read_csv('out/dist_matrix.csv')
        # distances = tf_idf.apply(cosine_sim, args=(tf_idf,), axis=1)
        predictions = knn_classifier(ground_truth, k, distances)
        temp = ground_truth[['author', 'file']]
        preds = pd.Series(list(predictions.values()))
        temp['prediction'] = preds
        temp[['file', 'prediction']].to_csv(f'\\results_KNN\\k_{k}.csv', index=False)


if __name__ == "__main__":
    main()
