from lifelines.fitters.kaplan_meier_fitter import KaplanMeierFitter
import matplotlib.pyplot as plt
import numpy as np
from consensus import ConsensusCluster
from cc2 import consensus_clustering
import collections
import pandas as pd
from sklearn.cluster import KMeans
import Cluster_Ensembles as CE
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA
from lifelines.statistics import multivariate_logrank_test
import seaborn as sns


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


def plot_km(c1, n, which, keep, fwrite, legend_l=[], xlim_top=120):
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
        if count >= keep:
            if not np.isnan(time_list1[i]):
                include.append(i)
    time1 = np.array(time_list1)[include]
    labels1 = np.array(cluster1.labels_)[include]
    surv1 = np.array(surv_list1)[include]

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
        if count >= keep:
            if not np.isnan(time_list1[i]):
                include.append(i)
    time2 = np.array(time_list1)[include]
    labels2 = np.array(cluster2.labels_)[include]
    surv2 = np.array(surv_list1)[include]
    if which == "us":
        time = time2
        labels = labels2
        surv = surv2
    else:
        time = time1
        labels = labels1
        surv = surv1
    kmf = KaplanMeierFitter()
    plt.clf()
    fig, ax = plt.subplots()
    # ax = plt.subplot(111)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    npat = len(labels)
    lab_l = np.unique(labels2)
    lab_l = np.sort(lab_l)
    if len(legend_l) == 0:
        ct = 0
        legend_l = {}
        for i in lab_l:
            legend_l[i] = "Subtype " + str(ct + 1)
            ct += 1
    for c in range(n):
        pindex = []
        for i in range(npat):
            if labels[i] == c:
                pindex.append(i)
        pindex = np.array(pindex)
        if len(pindex) == 0:
            continue
        print(time[pindex])
        print(surv[pindex])
        kmf.fit(
            list(time[pindex]),
            event_observed=list(surv[pindex]),
            label=legend_l[c] + " (n=" + str(len(pindex)) + ")",
        )
        kmf.plot(ax=ax, ci_show=False, linewidth=2.0)
        # print c,len(pindex),p2sur[pindex,0],p2sur[pindex,1]
    # print
    plt.ylim(0, 1)
    plt.legend(frameon=False, loc="lower left")
    plt.xlim(0, xlim_top)
    vals = ax.get_yticks()
    ax.set_yticklabels(["{:,.0%}".format(x) for x in vals])
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_aspect(abs(x1 - x0) / abs(y1 - y0))
    plt.ylabel("Survival rate", fontsize=20)
    plt.xlabel("Time (Months)", fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    if which == "base":
        title = "clustering without alignment, " + c1 + " K=" + str(n)
    else:
        title = c1 + " (p-value < 1e-3)"
    plt.title(title, fontsize=15)
    fig.tight_layout()
    fwrite = c1 + ".png"
    plt.savefig(fwrite, dpi=600)


plot_km(
    "LGG", 3, "us", 5, "km1.png", legend_l=[], xlim_top=60,
)
plot_km(
    "KIRC", 3, "us", 5, "km1.png", legend_l=[], xlim_top=60,
)

