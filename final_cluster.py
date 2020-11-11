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
import argparse
import os
from numpy.core.numeric import nan
from cc2 import consensus_clustering
import collections
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import Cluster_Ensembles as CE
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA
from lifelines.statistics import multivariate_logrank_test
import seaborn as sns

np.random.seed(1024)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target", type=str, default="LUAD", help="cancer type for target"
    )
    parser.add_argument(
        "--num_clusters", type=int, default=2, help="number of clusters"
    )
    opt = parser.parse_args()
    c1 = opt.target
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
    for index, row in name_mapping.iterrows():
        name = row["Approved symbol"]
        gene_id = row["NCBI Gene ID"]
        id2realname[gene_id] = name

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
    patients = []
    for index, row in c1_.iterrows():
        patient = row["id"]
        if patient in patient_data1:
            patients.append(patient)
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
    # time_list1 = np.nan_to_num(time_list1)
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
    c1_new = PCA(n_components=32, random_state=2).fit_transform(c1_data)
    dis_mat1 = euclidean_distances(c1_new, c1_new)
    min1 = 999
    min2 = 999
    n = opt.num_clusters
    p1 = 0
    p2 = 0
    min_index = np.argmin(sil_scores[n - 2])

    cur_score = sil_scores[n - 2]
    indexes = cur_score.argsort()[:5]

    cur_cluster = np.array(clusters)[indexes, n - 2, :].T
    # consensus = ConsensusCluster(KMeans, 2, 7, 1000)
    # consensus.fit(cur_cluster)
    # gen_labels = consensus.predict()
    cm = consensus_clustering(cur_cluster, KMeans(n_clusters=n, random_state=3))

    cluster2 = KMeans(n_clusters=n, random_state=3).fit(1 - cm)

    gen_labels = np.array(cluster2.labels_)
    results = np.zeros((len(patients), 2))
    results = results.astype("str")
    results[:, 0] = patients
    results[:, 1] = gen_labels.astype("str")
    np.savetxt(c1 + "_" + opt.num_clusters + ".txt", results, fmt="%s")


if __name__ == "__main__":
    main()
