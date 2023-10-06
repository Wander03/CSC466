import pandas as pd
import numpy as np
from sys import argv
from datetime import datetime

def check_row(data, i, n, c):
  if data.loc[i][c].sum(axis=0) == len(c):
    return 1/n
  else:
    return 0

def get_sup(data, n, minSup):
  sup = data.sum()/n
  if sup >= minSup:
    print(data.name, sup)
    return data.name
  return None

def candidateGen(F,k):
  C = []
  for i in F[k]:
    for j in F[k]:
      if (len(i) == k and len(j) == k):
        c = (i).union(j)
        flag = True
        if len(c) == k + 1:
          for l in range(k+1):
            s = list(c)[:l] + list(c)[l+1:]
            if set(s) not in F[k]:
              flag = False
          if flag == True:
            if c not in C:
              C += [c]
  return C

def getSkyline(F, k):
  F_skyline = []
  F.reverse()
  for f1 in F:
    if(not F_skyline):
      F_skyline.append(f1)
    else:
      flag = True
      for f2 in F_skyline:
        inter = f1.intersection(f2)
        if(len(inter) == len(f1)):
          flag = False
      if(flag):
        F_skyline.append(f1)

  return(F_skyline)

def Apriori(T, minSup):
  n = T.shape[0]
  F = {}
  F[1] = [{item} for item in list(T.apply(get_sup, args=(n, .1), axis=0).dropna())]
  F[2] = []
  k = 2
  while F[k-1] != []:
    C = candidateGen(F, k-1)
    sups = [0]*len(C)
    for i in range(n):
      for j in range(len(C)):
        sups[j] += check_row(T, i, n, list(C[j]))
    for l in range(len(C)):
      if sups[l] >= minSup:
        F[k] += [C[l]]
        print(sups[l])
    k += 1
    F[k] = []

    lst = list(F.values())
    flat = [item for sublist in lst for item in sublist]

  return getSkyline(flat, k-2)

def getConf(T, f, f_min_h):
  num = (T[f].sum(axis=1)==len(f)).sum()
  denom = (T[f_min_h].sum(axis=1)==len(f_min_h)).sum()
  return num/denom

def genRules(T,F,minConf):
  H1 = []
  for f in F:
    if len(f) >= 2:
      for i in range(len(f)):
        h = list(f)[i]
        f_min_h = list(f)[:i] + list(f)[i+1:]
        conf = getConf(T, list(f), f_min_h)
        if conf >= minConf:
          H1 += [str(f_min_h) + '-->' + str(h)]
  return H1

def out_goods(data, map_data, rules, arg):
  df_map = pd.read_csv(map_data)
  df_map["Flavor"] = df_map["Flavor"].str.replace("'", "")
  df_map["Food"] = df_map["Food"].str.replace("'", "")
  df_map["Item"] = df_map["Flavor"] + " " + df_map["Food"]
  id_item = df_map.set_index("Id")["Item"].to_dict()
  
  with open(data.split("\\")[-1] + "-out", "w") as f:
    f.write(f"Output for python3 {' '.join(arg)}\n\n")
    for i, r in enumerate(reversed(rules)):
      left = ", ".join([id_item.get(item-1, "Item") for item in r[0]])
      right = id_item.get(r[1]-1, "Item")

      f.write(f"Rule {i+1}:    {left} ---> {right}    [sup={round(r[2] * 100, 4)}, conf={round(r[3] * 100, 4)}]\n")

def out_bingo(data, df_map, rules, arg):
  id_item = df_map.set_index("Id")["Author(s)"].to_dict()

  with open(data.split("\\")[-1] + "-out", "w") as f:
    f.write(f"Output for python3 {' '.join(arg)}\n\n")
    for i, r in enumerate(reversed(rules)):
      left = ", ".join([id_item.get(item, "Author(s)") for item in r[0]])
      right = id_item.get(r[1], "Author(s)")

      f.write(f"Rule {i+1}:    {left} ---> {right}    [sup={round(r[2] * 100, 4)}, conf={round(r[3] * 100, 4)}]\n")

if __name__ == "__main__":
    print(datetime.now())

    data, minSup, minConf, data_map, goods = argv[1], float(argv[2]), float(argv[3]), argv[4], bool(int(argv[5]))

    if(goods):
      df = pd.read_csv(data, header=None)
      df.drop(df.columns[0], axis=1, inplace=True)
    else:
      df_map = pd.read_csv(data_map, sep="|", header=None, names=["Id", "Author(s)"])
      max_id = df_map["Id"].max()
      df = pd.DataFrame(columns=range(1, max_id+1))
      with open(data, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
          nums = [int(num.strip()) for num in line.split(",")][1:]
          new_row = pd.DataFrame([{n: 1 if n in nums else 0 for n in range(1, max_id+1)}])
          df = pd.concat([df, new_row], ignore_index=True)
        df.drop(df.columns[0], axis=1, inplace=True)

    skyline = Apriori(df, minSup)

    print(skyline)
    print(datetime.now())

    # rules = genRules(df, skyline, minConf)
# 
    # print(rules)
    # print(datetime.now())
# 
    # if(goods):
    #   out_goods(data, data_map, rules, argv)
    # else:
    #   out_bingo(data, df_map, rules, argv)
