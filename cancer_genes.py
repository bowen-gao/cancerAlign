import pandas as pd
import re
from scipy import stats
from sklearn.metrics import roc_auc_score
from scipy.stats import f_oneway


import numpy as np
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr

rpy2.robjects.numpy2ri.activate()

stats = importr("stats")

"""
data = pd.read_csv("cancer_gene_census.csv")
tumors =set()
data = data.fillna("")
for index, row in data.iterrows():
    gene = row["Gene Symbol"]
    tumor1 = row["Tumour Types(Somatic)"]
    tumor2 = row["Tumour Types(Germline)"]
    
    for tumor in tumor1.split(","):
        tumor = tumor.strip()
        if tumor not in tumors and tumor!="":
            tumors.add(tumor)
    for tumor in tumor2.split(","):
        tumor = tumor.strip()
        if tumor not in tumors and tumor!="":
            tumors.add(tumor)

tumors=list(tumors)
print(len(tumors),tumors)
import numpy as np
np.savetxt('tumors.txt',tumors,fmt="%s")
"""

data = pd.read_csv("cancer_genes_2.csv")
# data = data.iloc[1:, :]
# data.columns = data.iloc[0]
# print(data)
cancer_genes = {}
cancer_types = [
    "BRCA",
    "BLCA",
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
    "PRAD",
    "READ",
    "SARC",
    "STES",
    "TGCT",
    "THCA",
    "UCEC",
    "UVM",
]
for cancer in cancer_types:
    cancer_genes[cancer] = []
for index, row in data.iterrows():
    cancer = row["Cancer Type"]
    gene = row["Gene"]
    level = row["level"]
    if level == "A":
        if cancer == "ALL":
            for ct in cancer_types:
                cancer_genes[ct].append(gene)
        else:
            if cancer in cancer_genes:
                cancer_genes[cancer].append(gene)

import csv


for cancer in cancer_genes:
    genes = cancer_genes[cancer]
    genes = ",".join(genes)
    cancer_genes[cancer] = genes
with open("test.csv", "w") as f:
    writer = csv.writer(f)
    for row in cancer_genes.items():
        writer.writerow(row)


from consensus import ConsensusCluster
from cc2 import consensus_clustering
import collections
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import Cluster_Ensembles as CE
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA
from lifelines.statistics import multivariate_logrank_test

count = 0
s = 0
for c1 in cancer_types:
    df1 = pd.read_csv(
        "raw_survival/" + c1 + "_surv.txt_clean", index_col=None, sep="\t"
    )
    patient_data1 = {}
    for row in df1.values:
        patient_data1[row[0]] = [row[1], row[2]]

    df = pd.read_csv("new_data.csv", index_col=None)
    c1_ = df[df["cancer"] == c1]

    name_mapping = pd.read_csv("name_mapping.txt", sep="\t", dtype=str)
    id2realname = {}
    realname2id = {}
    for index, row in name_mapping.iterrows():
        name = row["Approved symbol"]
        gene_id = row["NCBI Gene ID"]
        id2realname[gene_id] = name
        realname2id[name] = gene_id

    """
    get feature gene name to gene index
    and gene index to gene name
    """

    name_2_index = {}
    index_2_name = {}

    num_genes = 0
    for entry in df["mut_genes"].values:
        names = entry.split(";")
        for name in names:
            if name not in name_2_index:
                name_2_index[name] = num_genes
                index_2_name[num_genes] = name
                num_genes += 1

    c1_count = {}
    time_list1 = []
    surv_list1 = []
    c1_data = []

    for index, row in c1_.iterrows():
        patient = row["id"]
        if patient in patient_data1:
            time_list1.append(patient_data1[patient][0])
            surv_list1.append(patient_data1[patient][1])
            genes = row["mut_genes"]
            genes = genes.split(";")
            vec = np.zeros(num_genes)
            for gene in genes:
                if gene in c1_count:
                    c1_count[gene] += 1
                else:
                    c1_count[gene] = 1
                vec[name_2_index[gene]] = 1
            c1_data.append(vec)

    c1_data = np.array(c1_data)
    gt_genes = cancer_genes[c1]
    gt_genes = gt_genes.split(",")
    gt_vec = np.zeros(c1_data.shape[1])
    for gene in gt_genes:
        if gene in realname2id:
            gene_id = realname2id[gene]
            if gene_id in name_2_index:
                index = name_2_index[gene_id]
                gt_vec[index] = 1
    clusters = []
    sil_scores = []
    for c2 in cancer_types:
        if c1 != c2:
            home_dir = "plots/" + c1 + "/" + c2 + "/"
            scores = np.loadtxt(home_dir + "scores")
            sil_scores.append(scores)
            # print(scores)
            labels = np.loadtxt(home_dir + "labels", delimiter=",")
            clusters.append(labels)
            # print(labels.shape)
    sil_scores = np.array(sil_scores).reshape((5, -1))

    for n in range(2, 7):
        inputs = []
        cur_score = sil_scores[n - 2]
        indexes = cur_score.argsort()[:5]

        cur_cluster = np.array(clusters)[indexes, n - 2, :].T
        # consensus = ConsensusCluster(KMeans, 2, 7, 1000)
        # consensus.fit(cur_cluster)
        # gen_labels = consensus.predict()
        cm = consensus_clustering(cur_cluster, KMeans(n_clusters=n, random_state=3))
        cluster2 = KMeans(n_clusters=n, random_state=3).fit(1 - cm)

        gen_labels = cluster2.labels_
        for i in range(n):
            index = np.argwhere(gen_labels == i)
            index = index.reshape(len(index))
            inputs.append(c1_data[index])
        if n == 2:
            t_stat_gen, pvalue_gen = stats.ttest_ind(inputs[0], inputs[1])
        if n == 3:
            t_stat_gen, pvalue_gen = f_oneway(inputs[0], inputs[1], inputs[2])
        if n == 4:
            t_stat_gen, pvalue_gen = f_oneway(
                inputs[0], inputs[1], inputs[2], inputs[3]
            )
        if n == 5:
            t_stat_gen, pvalue_gen = f_oneway(
                inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]
            )
        if n == 6:
            t_stat_gen, pvalue_gen = f_oneway(
                inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5]
            )

        pvalue_gen[np.isnan(pvalue_gen)] = 1

        pvalue_gen = pvalue_gen.astype(np.float32)

        if n == 2:
            large = np.mean(inputs[0], axis=0) > np.mean(inputs[1], axis=0)
            print(large.shape)
            temp = []
            for i in range(len(large)):
                if large[i]:
                    temp.append(1)
                else:
                    temp.append(pvalue_gen[i] / 2)
            pvalue_gen = np.array(temp)

        min_value = min(collections.Counter(gen_labels).values())
        if min_value < 5:
            continue
        else:
            break

    for n in range(2, 7):
        inputs = []
        c1_new = PCA(n_components=32, random_state=2).fit_transform(c1_data)
        dis_mat1 = euclidean_distances(c1_new, c1_new)
        cluster1 = KMeans(n_clusters=n, random_state=3).fit(dis_mat1)
        base_labels = cluster1.labels_
        for i in range(n):
            index = np.argwhere(base_labels == i)
            index = index.reshape(len(index))
            inputs.append(c1_data[index])
        if n == 2:
            t_stat_base, pvalue_base = stats.ttest_ind(inputs[0], inputs[1])
            print(inputs[0], inputs[1])
        if n == 3:
            t_stat_base, pvalue_base = f_oneway(inputs[0], inputs[1], inputs[2])
        if n == 4:
            t_stat_base, pvalue_base = f_oneway(
                inputs[0], inputs[1], inputs[2], inputs[3]
            )
        if n == 5:
            t_stat_base, pvalue_base = f_oneway(
                inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]
            )
        if n == 6:
            t_stat_base, pvalue_base = f_oneway(
                inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5]
            )

        pvalue_base[np.isnan(pvalue_base)] = 1

        pvalue_base = pvalue_base.astype(np.float32)
        if n == 2:
            large = np.mean(inputs[0], axis=0) > np.mean(inputs[1], axis=0)
            print(large.shape)
            temp = []
            for i in range(len(large)):
                if large[i]:
                    temp.append(1)
                else:
                    temp.append(pvalue_base[i] / 2)
            pvalue_base = np.array(temp)
        min_value = min(collections.Counter(base_labels).values())
        if min_value < 5:
            continue
        else:
            break
    print(
        c1,
        roc_auc_score(gt_vec, 1 - pvalue_base),
        roc_auc_score(gt_vec, 1 - pvalue_gen),
    )
    s += roc_auc_score(gt_vec, 1 - pvalue_gen)
    with open("new2.csv", "a") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                c1,
                roc_auc_score(gt_vec, 1 - pvalue_base),
                roc_auc_score(gt_vec, 1 - pvalue_gen),
            ]
        )

    if roc_auc_score(gt_vec, 1 - pvalue_base) < roc_auc_score(gt_vec, 1 - pvalue_gen):
        count += 1
print(count, s)

