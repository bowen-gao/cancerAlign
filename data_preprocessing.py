import pandas as pd
from _collections import defaultdict
import numpy as np

cancer_types = [
    "BLCA",
    "BRCA",
    "CESC",
    "CHOL",
    "COAD",
    "DLBC",
    "GBM",
    "HNSC",
    "KICH",
    "KIRC",
    "LGG",
    "LIHC",
    "LUAD",
    "LUSC",
    "OV",
    "PAAD",
    "PCPG",
    "PRAD",
    "READ",
    "SARC",
    "STAD",
    "STES",
    "TGCT",
    "THCA",
    "UCEC",
    "UCS",
    "UVM",
]

id_2_genes = defaultdict(list)
id_2_cancer = defaultdict(str)
id_2_stage = defaultdict(str)
genes = set()
for cancer in cancer_types:
    file_name = "raw_survival/" + cancer + "_mut.txt"
    df = pd.read_csv(file_name, sep="\t")
    ids = df.iloc[:, 0].values
    mut_genes = df.iloc[:, 1].values
    for i, id in enumerate(ids):
        id_2_genes[id].append(mut_genes[i])
        genes.add(mut_genes[i])
        id_2_cancer[id] = cancer
print(id_2_cancer)
print(len(list(genes)))
print(max(genes))
num = []
for id in id_2_genes:
    num.append(len(id_2_genes[id]))
print(np.mean(np.array(num)))


data = {"id": [], "cancer": [], "mut_genes": []}

for id in id_2_cancer:
    if id in id_2_genes:
        data["id"].append(id)
        data["cancer"].append(id_2_cancer[id])
        genes = [str(gene) for gene in id_2_genes[id]]
        data["mut_genes"].append(";".join(genes))

df = pd.DataFrame(data, columns=["id", "cancer", "mut_genes"])
print(df)
df.to_csv("new_data.csv", index=None)

