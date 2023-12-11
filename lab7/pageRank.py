"""
Course: CSC 466
Quarter: Fall 2023
Assigment: Lab 7

Name(s):
    Andrew Kerr // adkerr@calpoly.edu

Description:
    How to run:
        python3 pageRank.py
            <datafile>
            <OPTIONAL: damping factor - default = 0.85>
"""
import pandas as pd
import numpy as np
from sys import argv
import time


def convert_overtime(x):
    return bool(x and x != "(OT)")

def adjacency_matrix(df):
    unique_nodes = np.unique(np.concatenate((df['col1'].unique(), df['col3'].unique())))
    adjacency_matrix = pd.DataFrame(0, index=unique_nodes, columns=unique_nodes, dtype=int)
    try:
        for index, row in df.iterrows():
            node1, score1, node2, score2 = row
            if score1 > score2:
                adjacency_matrix.loc[node2, node1] = 1
            else:
                adjacency_matrix.loc[node1, node2] = 1
    except:
        for index, row in df.iterrows():
            node1, score1, node2, score2, _ = row
            if score1 > score2:
                adjacency_matrix.loc[node2, node1] = 1
            else:
                adjacency_matrix.loc[node1, node2] = 1
    
    return adjacency_matrix

def transition_matrix(adjacency_matrix, d):
    n = adjacency_matrix.shape[0]
    out_links = adjacency_matrix.sum(axis=0).replace(0, 1)
    transition_matrix = (d * (adjacency_matrix.T / out_links)).T + (1 - d) / n
    return transition_matrix

def page_rank(transition_matrix):
    n = transition_matrix.shape[0]
    pi = np.ones(n) / n
    old_pi = pi.copy()

    for i in range(10000):
        pi = transition_matrix @ pi
        if np.allclose(pi, old_pi, atol=1e-6):
            print('Converged!')
            return pi, i
        old_pi = pi.copy()
    
    return pi, i

def main():
    start_read = time.time()
    try:
        column_names = ['col1', 'col2', 'col3', 'col4']
        columns_type = {'col1': str, 'col2': int, 'col3': str, 'col4': int}
        data = pd.read_csv(argv[1], names=column_names, dtype=columns_type)
    except:
        column_names = ['col1', 'col2', 'col3', 'col4', 'col5']
        converters = {'col5': convert_overtime}
        columns_type = {'col1': str, 'col2': int, 'col3': str, 'col4': int}
        data = pd.read_csv(argv[1], delimiter=', ', names=column_names, dtype=columns_type, usecols=range(5), converters=converters, engine='python')
        data['col1'] = data['col1'].str.strip('"')
        data['col3'] = data['col3'].str.strip('"')

    if len(argv) == 3:
        d = float(argv[2])
    else:
        d = 0.85

    am = adjacency_matrix(data)
    tm = transition_matrix(am, d)
    end_read = time.time()

    start_processing = time.time()
    rank, iterations = page_rank(tm)
    end_processing = time.time()

    df_rank = pd.DataFrame({'PageRank': rank})
    df_rank['Rank'] = df_rank['PageRank'].rank(ascending=False)
    df_rank = df_rank[['Rank', 'PageRank']]
    df_rank.sort_values(by='Rank', inplace=True)

    out = argv[1].split("/")[-1] if "/" in argv[1] else argv[1].split("\\")[-1]
    with open(f'results\\{out[:-4]}_{d}.txt', 'w') as f:
        stdout = f
        print('Read Time:', end_read - start_read, file=stdout)
        print('Processing Time:', end_processing - start_processing, file=stdout)
        print('Number of Iterations:', iterations, file=stdout)
        print(file=stdout)
        print('Damping Factor:', d, file=stdout)
        print(file=stdout)
        print(f"{'Rank':<8}{'Node':<30}{'Page Rank':<20}", file=stdout)
        for index, row in df_rank.iterrows():
            print(f"{row['Rank']:<8.1f}{row.name:<30}{row['PageRank']:<20.18f}", file=stdout)

    print('Read Time:', end_read - start_read)
    print('Processing Time:', end_processing - start_processing)
    print('Number of Iterations:', iterations)

    count = 1
    print(f"{'Rank':<8}{'Node':<30}{'Page Rank':<20}")
    for index, row in df_rank.iterrows():
        if count == 16:
            break
        print(f"{row['Rank']:<8.1f}{row.name:<30}{row['PageRank']:<20.18f}")
        count += 1

if __name__ == "__main__":
    main()
