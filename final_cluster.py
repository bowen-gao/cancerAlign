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
from numpy.core.numeric import nan
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
import seaborn as sns

np.random.seed(1024)


def cluster(c1):
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
    for n in range(2, 7):
        cluster1 = KMeans(n_clusters=n, random_state=3).fit(dis_mat1)
        """
        mat = []
        for labels in clusters:
            gen_labels = labels[n - 2]
            mat.append(gen_labels)
        mat = np.array(mat)
        ce_labels = CE.cluster_ensembles(mat)
        """
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

        gen_labels = cluster2.labels_
        d = collections.Counter(cluster1.labels_)
        include = []
        for i in range(len(time_list1)):
            label = cluster1.labels_[i]
            count = d[label]
            if count >= 5:
                if not np.isnan(time_list1[i]):
                    include.append(i)
        results1 = multivariate_logrank_test(
            list(np.array(time_list1)[include]),
            list(np.array(cluster1.labels_)[include]),
            event_observed=list(np.array(surv_list1)[include]),
        )
        p1 = results1.p_value
        if np.isnan(p1):
            p1 = 1
        if p1 < min1 and p1 > 1e-8:
            min1 = p1
        d = collections.Counter(gen_labels)
        include = []
        for i in range(len(time_list1)):
            label = gen_labels[i]
            count = d[label]
            if count >= 5:
                if not np.isnan(time_list1[i]):
                    include.append(i)
        results2 = multivariate_logrank_test(
            list(np.array(time_list1)[include]),
            list(np.array(gen_labels)[include]),
            event_observed=list(np.array(surv_list1)[include]),
        )
        p2 = results2.p_value
        if np.isnan(p2):
            p2 = 1
        if p2 < min2 and p2 > 1e-8:
            min2 = p2
        # print(c1, n, results1.p_value, results2.p_value)
    print(c1, min1, min2)
    return min1, min2


def cluster_n(c1, n):
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
    cluster1 = KMeans(n_clusters=n, random_state=3).fit(dis_mat1)
    d = collections.Counter(cluster1.labels_)
    include = []
    for i in range(len(time_list1)):
        label = cluster1.labels_[i]
        count = d[label]
        if count >= 5:
            if not np.isnan(time_list1[i]):
                include.append(i)
    results1 = multivariate_logrank_test(
        list(np.array(time_list1)[include]),
        list(np.array(cluster1.labels_)[include]),
        event_observed=list(np.array(surv_list1)[include]),
    )
    p1 = results1.p_value

    cur_score = sil_scores[n - 2]
    indexes = cur_score.argsort()[:5]
    cur_cluster = np.array(clusters)[indexes, n - 2, :].T
    cm = consensus_clustering(cur_cluster, KMeans(n_clusters=n, random_state=3)) / 5
    cluster2 = KMeans(n_clusters=n, random_state=3).fit(1 - cm)
    gen_labels = cluster2.labels_
    d = collections.Counter(gen_labels)
    include = []
    for i in range(len(time_list1)):
        label = gen_labels[i]
        count = d[label]
        if count >= 5:
            if not np.isnan(time_list1[i]):
                include.append(i)

    results2 = multivariate_logrank_test(
        list(np.array(time_list1)[include]),
        list(np.array(gen_labels)[include]),
        event_observed=list(np.array(surv_list1)[include]),
    )
    p2 = results2.p_value
    # print(c1, n, results1.p_value, results2.p_value)
    if np.isnan(p2):
        p2 = 1
    if np.isnan(p1):
        p1 = 1
    print(p1, p2)
    return p1, p2


count = 0
import matplotlib.pyplot as plt


for n in range(2, 3):
    plt.clf()
    plt.xlim(1, 1e-5)
    plt.ylim(1, 1e-5)
    xpoints = ypoints = plt.xlim()
    newx = (1.0, 0.01)
    plt.plot(xpoints, xpoints, "--", color="salmon")
    plt.plot(newx, [0.01] * len(xpoints), "--", color="gray")
    plt.plot([0.01] * len(xpoints), newx, "--", color="gray")
    for c1 in cancer_types:
        min1, min2 = cluster_n(c1, n)
        plt.scatter(min2, min1, color="#377FB8")
        if min1 < 0.01 or min2 < 0.01:
            if c1 == "BLCA":
                plt.annotate(
                    c1, (np.exp(np.log(min2) - 0.2), np.exp(np.log(min1) + 0.2))
                )
            elif c1 == "LGG":
                plt.annotate(
                    c1, (np.exp(np.log(min2) - 0.2), np.exp(np.log(min1) + 0.2))
                )
            elif c1 == "KICH":
                plt.annotate(
                    c1, (np.exp(np.log(min2) + 0.2), np.exp(np.log(min1) + 0.6))
                )
            else:
                plt.annotate(c1, (min2, np.exp(np.log(min1) - 0.2)))
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("Survival association $\it{p}$-value\ncancerAlign", fontsize=16)
    plt.ylabel(
        "clustering without alignment\nSurvival association $\it{p}$-value", fontsize=16
    )
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    sns.despine()
    sns.despine(top=True, right=True, left=False, bottom=False)
    plt.savefig(str(n) + "_test.png", dpi=600)
"""

plt.xlim(1, 1e-8)
plt.ylim(1, 1e-8)
xpoints = ypoints = plt.xlim()
newx = (1.0, 0.01)
plt.plot(xpoints, xpoints, "--", color="salmon")
plt.plot(newx, [0.01] * len(xpoints), "--", color="gray")
plt.plot([0.01] * len(xpoints), newx, "--", color="gray")
for c1 in cancer_types:
    min1, min2 = cluster(c1)

    plt.scatter(min2, min1, color="#377FB8")
    if min1 < 0.01 or min2 < 0.01:
        if c1 == "HNSC":
            plt.annotate(c1, (min2, np.exp(np.log(min1) - 0.4)))
        elif c1 == "SARC":
            plt.annotate(c1, (np.exp(np.log(min2) - 0.4), np.exp(np.log(min1) + 0.2)))
        elif c1 == "DLBC":
            plt.annotate(c1, (np.exp(np.log(min2) - 0.3), np.exp(np.log(min1) + 0.15)))
        elif c1 == "READ":
            plt.annotate(c1, (np.exp(np.log(min2) - 0.2), np.exp(np.log(min1) + 0.3)))
        else:
            plt.annotate(c1, (min2, np.exp(np.log(min1) - 0.2)))
plt.yscale("log")
plt.xscale("log")
plt.xlabel("Survival association $\it{p}$-value\ncancerAlign", fontsize=16)
plt.ylabel(
    "clustering without alignment\nSurvival association $\it{p}$-value", fontsize=16
)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
sns.despine()
sns.despine(top=True, right=True, left=False, bottom=False)
plt.savefig("test.png", dpi=600)

"""
