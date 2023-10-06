import pandas as pd
from sys import argv

def get_subsets(c):
  s = []
  c = list(c)
  for i in range(len(c)):
    s.append(set(c[:i] + c[i+1:]))
  return(s)

def count_row(row, C, c_count):
  for i, c in enumerate(C):
    if(row[list(c)].sum() == len(c)):
      c_count[i] += 1

def get_initial_freq(col, n, minSup, F):
  freq = col.sum() / n
  if(freq >= minSup):
    F[1].append(({col.name}, freq))

def get_skyline(F, k):
  F_skyline = []
  F.reverse()
  for f1 in F:
    if(not F_skyline):
      F_skyline.append(f1)
    else:
      flag = True
      for f2 in F_skyline:
        inter = f1[0].intersection(f2[0])
        if(len(inter) == len(f1[0])):
          flag = False
      if(flag):
        F_skyline.append(f1)

  return(F_skyline)

def candidateGen(F, k):
  C = []
  for i in F[k]:
    for j in F[k]:
      c = i[0].union(j[0])
      if(len(c) == k + 1 and c not in C):
        if any(s in (f[0] for f in F[k]) for s in get_subsets(c)):
          C.append(c)

  return(C)

def Apriori(df, minSup):
  """
  df: Pandas DataFrame of binary market baskets
  minSup: a float
  """
  n = len(df)
  k = 2
  F = {1: []}
  df.apply(lambda col: get_initial_freq(col, n, minSup, F))

  while(len(F[k-1]) != 0):
    C = candidateGen(F, k-1)
    c_count = [0] * len(C)
    df.apply(lambda row: count_row(row, C, c_count), axis=1)
    F.update({k: [(c, c_count[i]/n) for i, c in enumerate(C) if c_count[i]/n >= minSup]})
    k += 1

  return(get_skyline([f for s in F.values() for f in s], k-2))

def genRules(df, F, minConf):
  """
  df: Pandas DataFrame of binary market baskets
  F: dictionary of frequent itemsets
  minConf: a float
  """
  H = []
  for f in F:
    if(len(f[0]) >= 2):
      for s in f[0]:
        temp_f = f[0].copy()
        sup_n = df[list(temp_f)].apply(lambda row: 1 if sum(row) == len(row) else 0, axis=1).sum()
        temp_f.remove(s)
        sup_d = df[list(temp_f)].apply(lambda row: 1 if sum(row) == len(row) else 0, axis=1).sum()
        if(sup_n / sup_d >= minConf):
          H += [([temp_f, s, sup_n / sup_d])]

  return(H)

if __name__ == "__main__":
    data, minSup, minConf = argv[1], argv[2], argv[3]
    df = pd.read_csv(data, header=False)
    skyline = Apriori(df, minSup)
    rules = genRules(df, skyline, minConf)
    print(rules)
