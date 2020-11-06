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
from torch import dtype

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch


class Generator(nn.Module):
    def __init__(self, num_genes):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(num_genes, 128, normalize=False),
            # *block(512, 512),
            # *block(512, 512),
            nn.Linear(128, num_genes),
            nn.Sigmoid()
        )

    def forward(self, z):
        out = self.model(z)

        return out


def cluster_n(c1, c2, c3, n):
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
    print(c1_data.shape)

    c1_new = PCA(n_components=32, random_state=2).fit_transform(c1_data)
    dis_mat1 = euclidean_distances(c1_new, c1_new)
    cluster1 = KMeans(n_clusters=n, random_state=3).fit(dis_mat1)
    d = collections.Counter(cluster1.labels_)
    label, count = min(d.items(), key=lambda x: x[1])
    include = []
    for i in range(len(time_list1)):
        label = cluster1.labels_[i]
        count = d[label]
        if count >= 5:
            include.append(i)
    results1 = multivariate_logrank_test(
        list(np.array(time_list1)[include]),
        list(np.array(cluster1.labels_)[include]),
        event_observed=list(np.array(surv_list1)[include]),
    )
    p1 = results1.p_value

    generator1 = Generator(num_genes)
    PATH = "all_models/" + c1 + "_" + c2 + "/" + str(n) + ".pt"
    generator1.load_state_dict(torch.load(PATH))
    generator1.cuda()
    generator1.eval()
    generator2 = Generator(num_genes)
    PATH = "all_models/" + c2 + "_" + c3 + "/" + str(n) + ".pt"
    generator2.load_state_dict(torch.load(PATH))
    generator2.cuda()
    generator2.eval()
    c1_data = torch.cuda.FloatTensor(c1_data)
    c2_data = generator2(generator1(c1_data))
    c2_data = c2_data.cpu().detach().numpy()
    c2_new = PCA(n_components=32, random_state=2).fit_transform(c2_data)
    dis_mat2 = euclidean_distances(c2_new, c2_new)
    cluster2 = KMeans(n_clusters=n, random_state=3).fit(dis_mat2)
    gen_labels = cluster2.labels_
    d = collections.Counter(gen_labels)
    include = []
    for i in range(len(time_list1)):
        label = gen_labels[i]
        count = d[label]
        if count >= 5:
            include.append(i)
    results2 = multivariate_logrank_test(
        list(np.array(time_list1)[include]),
        list(np.array(gen_labels)[include]),
        event_observed=list(np.array(surv_list1)[include]),
    )
    p2 = results2.p_value
    home_dir = "plots/" + c1 + "/" + c2 + "/"
    labels = np.loadtxt(home_dir + "labels", delimiter=",")
    cancergan_label = labels[n - 2]
    d = collections.Counter(cancergan_label)
    include = []
    for i in range(len(time_list1)):
        label = cancergan_label[i]
        count = d[label]
        if count >= 5:
            include.append(i)
    results3 = multivariate_logrank_test(
        list(np.array(time_list1)[include]),
        list(np.array(cancergan_label)[include]),
        event_observed=list(np.array(surv_list1)[include]),
    )
    p3 = results3.p_value

    return p1, p2, p3


count = 0
import matplotlib.pyplot as plt


for n in range(2, 7):
    min1, min2, min3 = cluster_n("HNSC", "LGG", "LUSC", n)
    print(min1, min2, min3)

"""
plt.xlim(1, 1e-8)
plt.ylim(1, 1e-8)
xpoints = ypoints = plt.xlim()
newx = (1.0, 0.01)
plt.plot(xpoints, xpoints, "--", color="gray")
plt.plot(newx, [0.01] * len(xpoints), "--", color="gray")
plt.plot([0.01] * len(xpoints), newx, "--", color="gray")
for c1 in cancer_types:
    min1, min2 = cluster(c1)
    plt.scatter(min2, min1)
    plt.annotate(c1, (min2, min1))
plt.yscale("log")
plt.xscale("log")

plt.savefig("test.png")
"""

