import pandas as pd
import re
from scipy import stats
from sklearn.metrics import roc_auc_score
from scipy.stats import f_oneway
from scipy.stats import chi2_contingency

import scipy.stats as stats


data = pd.read_csv("known_cancer_genes.csv")
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
        elif cancer == "COAD/READ":
            cancer_genes["READ"].append(gene)
            cancer_genes["COAD"].append(gene)
        else:
            if cancer in cancer_genes:
                cancer_genes[cancer].append(gene)

import csv


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
import argparse

count = 0
s = 0


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target", type=str, default="LUAD", help="cancer type for target"
    )
    opt = parser.parse_args()
    c1 = opt.target
    n = 2
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
    gt_vec = np.zeros(c1_data.shape[1])
    for gene in gt_genes:
        if gene in realname2id:
            gene_id = realname2id[gene]
            if gene_id in name_2_index:
                index = name_2_index[gene_id]
                gt_vec[index] = 1
    clusters = []
    base_clusters = []
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

    cur_score = sil_scores[n - 2]
    indexes = cur_score.argsort()[:5]

    cur_cluster = np.array(clusters)[indexes, n - 2, :].T
    # consensus = ConsensusCluster(KMeans, 2, 7, 1000)
    # consensus.fit(cur_cluster)
    # gen_labels = consensus.predict()
    cm = consensus_clustering(cur_cluster, KMeans(n_clusters=n, random_state=3))
    cluster2 = KMeans(n_clusters=n, random_state=3).fit(1 - cm)

    gen_labels = cluster2.labels_

    new_data = c1_data.T
    pvalue_gen = []
    for i in range(len(new_data)):
        conti = np.zeros((n, 2))
        for j in range(len(new_data[i])):
            conti[int(gen_labels[j])][int(new_data[i][j])] += 1
        # odd, p = stats.fisher_exact(conti)
        try:
            _, p, _, _ = chi2_contingency(conti)
        except:
            p = 1
        pvalue_gen.append(p)

    pvalue_gen = np.array(pvalue_gen)

    # pvalue_gen[np.isnan(pvalue_gen)] = 1

    pvalue_gen = pvalue_gen.astype(np.float32)
    dic = {}
    for i in range(len(pvalue_gen)):
        id_name = index_2_name[i]
        if id_name in id2realname:
            real_name = id2realname[id_name]
            dic[real_name] = pvalue_gen[i]

    print(sorted(dic, key=dic.get)[:10])


if __name__ == "__main__":
    main()

