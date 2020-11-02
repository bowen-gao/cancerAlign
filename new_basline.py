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


def cluster(c1, c2):
    df1 = pd.read_csv(
        "raw_survival/" + c1 + "_surv.txt_clean", index_col=None, sep="\t"
    )
    patient_data1 = {}
    for row in df1.values:
        patient_data1[row[0]] = [row[1], row[2]]

    df = pd.read_csv("new_data.csv", index_col=None)
    c1_ = df[df["cancer"] == c1]
    c2_ = df[df["cancer"] == c2]

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
    for entry in c1_["mut_genes"].values:
        names = entry.split(";")
        for name in names:
            if name not in name_2_index:
                name_2_index[name] = num_genes
                index_2_name[num_genes] = name
                num_genes += 1
    for entry in c2_["mut_genes"].values:
        names = entry.split(";")
        for name in names:
            if name not in name_2_index:
                name_2_index[name] = num_genes
                index_2_name[num_genes] = name
                num_genes += 1
    print(num_genes)
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
    print(c1_data.shape)
    c2_data = []
    c2_count = {}
    for index, row in c2_.iterrows():
        genes = row["mut_genes"]
        genes = genes.split(";")
        vec = np.zeros(num_genes)
        for gene in genes:
            if gene in c2_count:
                c2_count[gene] += 1
            else:
                c2_count[gene] = 1
            vec[name_2_index[gene]] = 1
        c2_data.append(vec)

    all_data = np.concatenate((c1_data, c2_data), axis=0)
    print(all_data.shape)
    c1_new = PCA(n_components=32, random_state=2).fit_transform(all_data)
    dis_mat1 = euclidean_distances(c1_new, c1_new)
    labels = []
    for n in range(2, 7):
        cluster1 = KMeans(n_clusters=n, random_state=3).fit(dis_mat1)
        labels.append(cluster1.labels_[: len(c1_data)])
    labels = np.array(labels)
    print(labels.shape)
    np.savetxt(
        "baseline_labels/" + c1 + "_" + c2, labels.astype(int), fmt="%i", delimiter=","
    )


for c1 in cancer_types:
    for c2 in cancer_types:
        if c1 != c2:
            cluster(c1, c2)

