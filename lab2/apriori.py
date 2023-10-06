import pandas as pd
from sys import argv
from datetime import datetime

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
          H.append([([temp_f, s, f[1], sup_n / sup_d])])

  return(H)

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

    rules = genRules(df, skyline, minConf)

    print(rules)
    print(datetime.now())

    if(goods):
      out_goods(data, data_map, rules, argv)
    else:
      out_bingo(data, df_map, rules, argv)
