"""
Course: CSC 466
Quarter: Fall 2023
Assigment: Lab 5

Name(s):
    Brendan Callender // bscallen@calpoly.edu
    Andrew Kerr // adkerr@calpoly.edu

Description:
    How to run: python3 EvaluateCFList.py
                    <jester-data-1.csv: Path to jester-data-1.csv file>
                    <Method: Memory-based method to use>
                       Valid Methods:
                            1: Weighted Sum (User Based)
                            2: Weighted Sum (Item Based)
                            3: Adj Weighted Sum (User Based)
                            4: Adj Weighted Sum (Item Based)
                            5: Weighted Sum w/ KNN (User Based)
                            6: Adj Weighted Sum w/ KNN (User Based)
                    <Filename: File containing list of test cases>
                    <OPTIONAL: KNN: Number of Nearest Neighbors to use>
"""
import sys
import pandas as pd
import numpy as np


def cosine_sim_users(d1, d2):
    return d2.apply(lambda row: np.dot(d1.to_numpy(), row.to_numpy()) / (np.linalg.norm(d1.to_numpy()) * np.linalg.norm(row.to_numpy())), axis=1)


def cosine_sim_items(d1, d2):
    return d2.apply(lambda row: np.dot(d1.to_numpy(), row.to_numpy()) / (np.linalg.norm(d1.to_numpy()) * np.linalg.norm(row.to_numpy())), axis=0)


# Method 1: Weighted Sum (User Based)
def calc_weighted_sum_ub(u, i, D, user_sims, KNN=0):
    if KNN > 0:
        knn_idx = user_sims.drop(u).sort_values(ascending=False)[:KNN].index
        util = D.iloc[knn_idx, i]
        sims = user_sims[knn_idx]
        k = 1 / np.abs(sims).sum()
        return k*(util*sims).sum()
    else:
        util = D.iloc[:, i].drop(u)
        sims = user_sims.drop(u)
        k = 1 / np.abs(sims).sum()
        return k*(util*sims).sum()


# Method 2: Adjusted Weighted Sum (User Based)
def calc_adj_weighted_sum_ub(u, i, D, user_sims, user_means, KNN=0):
    if KNN > 0:
        knn_idx = user_sims.drop(u).sort_values(ascending=False)[:KNN].index
        adj_util = D.iloc[knn_idx, i] - user_means[knn_idx]
        sims = user_sims[knn_idx]
        k = 1 / np.abs(sims).sum()
        return user_means[u] + k*(sims*(adj_util)).sum()
    else:
        adj_util = D.iloc[:, i].drop(u) - user_means.drop(u)
        sims = user_sims.drop(u)
        k = 1 / np.abs(sims).sum()
        return user_means[u] + k*(sims*(adj_util)).sum()


# Method 3: Weighted Sum (Item Based)
def calc_weighted_sum_ib(u, i, D, item_sims):
    util = D.iloc[u,:].drop(i)
    sims = item_sims.drop(i)
    k = 1 / np.abs(sims).sum()
    return k*(util*sims).sum()


# Method 4: Adjusted Weighted Sum (Item Based)
def calc_adj_weighted_sum_ib(u, i, D, item_sims, item_means):
    adj_util = D.iloc[u,:].drop(i) - item_means.drop(i)
    sims = item_sims.drop(i)
    k = 1 / np.abs(sims).sum()
    return item_means[i] + k*(sims*(adj_util)).sum()


def calc_rating(u, i, D, D_prime, D_NaN, user_means, item_means, method, KNN=0):
    if method in [1, 3, 5, 6]:
        sims = D_prime.loc[u:u, :].apply(cosine_sim_users, args=(D_prime,), axis=1).loc[u]
    elif method in [2, 4]:
        sims = D_prime.loc[:, i:i].apply(cosine_sim_items, args=(D_prime,), axis=0)[i]

    if method == 1:
        return calc_weighted_sum_ub(u, i, D_NaN, sims)
    elif method == 2:
        return calc_weighted_sum_ib(u, i, D_NaN, sims)
    elif method == 3:
        return calc_adj_weighted_sum_ub(u, i, D_NaN, sims, user_means)
    elif method == 4:
        return calc_adj_weighted_sum_ib(u, i, D_NaN, sims, item_means)
    elif method == 5:
        return calc_weighted_sum_ub(u, i, D_NaN, sims, KNN=KNN)
    elif method == 6:
        return calc_adj_weighted_sum_ub(u, i, D_NaN, sims, user_means, KNN=KNN)
    else:
        print('Please Pick Valid Method')
        print('1: Weighted Sum (User Based)')
        print('2: Weighted Sum (Item Based)')
        print('3: Adj Weighted Sum (User Based)')
        print('4: Adj Weighted Sum (Item Based)')
        print('5: Weighted Sum w/ KNN (User Based)')
        print('6: Adj Weighted Sum w/ KNN (User Based)')


def eval_list(D, method, filename, KNN):
    res = pd.DataFrame(columns=['userID', 'itemID', 'Actual_Rating', 'Predicted_Rating', 'Delta_Rating'])

    actual = []
    predicted = []

    D_prime = D.replace(99, 0)
    D_NaN = D.replace(99, None)

    item_means = D_NaN.mean(axis=0)
    user_means = D_NaN.mean(axis=1)

    test_cases = pd.read_csv(filename, header=None)

    num = 0
    for i in range(test_cases.shape[0]):

        u = test_cases.iloc[i, 0]
        i = test_cases.iloc[i, 1]

        val = D.iloc[u, i]
        if val != 99:
            num += 1

            pred = calc_rating(u, i, D, D_prime, D_NaN, user_means, item_means, method, KNN)

            actual.append(val)
            predicted.append(pred)

            y = pd.DataFrame(
                {'userID': u, 'itemID': i, 'Actual_Rating': val, 'Predicted_Rating': pred, 'Delta_Rating': val - pred},
                index=[num - 1])
            res = pd.concat([res, y], axis=0)

    actual = pd.Series(actual)
    predicted = pd.Series(predicted)

    mae = (np.abs(actual - predicted)).mean()

    rec_actual = (actual >= 5).astype(int)
    rec_predicted = (predicted >= 5).astype(int)

    tp = (rec_actual[rec_actual == 1] == rec_predicted[rec_actual == 1]).sum()
    fp = (rec_actual[rec_actual == 0] != rec_predicted[rec_actual == 0]).sum()
    tn = (rec_actual[rec_actual == 0] == rec_predicted[rec_actual == 0]).sum()
    fn = (rec_actual[rec_actual == 1] != rec_predicted[rec_actual == 1]).sum()

    conf_matrix = pd.DataFrame([[tp, fn], [fp, tn]], columns=['Rec', 'No Rec'], index=['Rel', 'Not Rel'])

    if tp + fp == 0:
        prec = 0
    else:
        prec = tp / (tp + fp)
    if tp + fn == 0:
        rec = 0
    else:
        rec = tp / (tp + fn)
    if prec + rec == 0:
        f1 = 0
    else:
        f1 = 2 * (prec * rec) / (prec + rec)
    acc = (tp + tn) / (tp + fp + tn + fn)

    print(res)
    print()
    print(conf_matrix)
    print()
    print('Metrics:')
    print(f'MAE: {mae}')
    print(f'Accuracy: {acc}')
    print(f'Precision: {prec}')
    print(f'Recall: {rec}')
    print(f'F1-Score: {f1}')
    print()


def main():
    args = sys.argv

    if len(args) < 3:
        print("Syntax: python3 EvaluateCFList.py <jester-data-1.csv> <Method> <Filename> <OPTIONAL: KNN>")

        print('Valid Methods:')
        print('1: Weighted Sum (User Based)')
        print('2: Weighted Sum (Item Based)')
        print('3: Adj Weighted Sum (User Based)')
        print('4: Adj Weighted Sum (Item Based)')
        print('5: Weighted Sum w/ KNN (User Based)')
        print('\tSpecify KNN parameter')
        print('6: Adj Weighted Sum w/ KNN (User Based)')
        print('\tSpecify KNN parameter')
        return None

    method = int(args[2])
    infile = args[3]

    if method == 5 or method == 6:
        try:
            KNN = int(args[4])
        except:
            print('Specify KNN Parameter')
            return None

    D = pd.read_csv(args[1], header=None)
    eval_list(D, method, infile, KNN)

if __name__ == "__main__":
    main()