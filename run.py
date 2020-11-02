# used to implement the new GAN model which is used to improve the patient clustering
# plan: train c1 -> c2, use c2_generated to do clustering, remove c2, and others are clustering for c1
# try knn, k= = 2,3,4,5,6 with and without gan. compare results
# log rank test: l1: survival time for patients  l2: clustering class for patients
# DBSCAN, KNN

import argparse
import os
from sys import flags
import numpy as np
import math
import pandas as pd
from torch import dtype

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
from sklearn.preprocessing import Normalizer
from lifelines.statistics import multivariate_logrank_test

import scipy.stats as stats

import collections
import matplotlib.pyplot as plt

"""
This code is adapted from the PyTorch GAN implemetations:
https://github.com/eriklindernoren/PyTorch-GAN
"""

"""
KIRC late -> STAD late (train)
KIRC early -> STAD early (evaluation)
"""

from csv import writer


def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, "a+", newline="") as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)


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


class Discriminator(nn.Module):
    def __init__(self, num_genes):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(num_genes, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, gene_exp):
        validity = self.model(gene_exp)

        return validity


from sklearn.cluster import KMeans


def get_cluster(X, n):
    kmeans = KMeans(n_clusters=n, random_state=0).fit(X)


def main(c1, c2):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_epochs", type=int, default=100, help="number of epochs of training"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="size of the batches"
    )
    parser.add_argument("--lr", type=float, default=2e-5, help="adam: learning rate")
    parser.add_argument(
        "--b1",
        type=float,
        default=0.9,
        help="adam: decay of first order momentum of gradient",
    )
    parser.add_argument(
        "--b2",
        type=float,
        default=0.999,
        help="adam: decay of first order momentum of gradient",
    )
    parser.add_argument(
        "--n_cpu",
        type=int,
        default=8,
        help="number of cpu threads to use during batch generation",
    )
    parser.add_argument(
        "--latent_dim", type=int, default=100, help="dimensionality of the latent space"
    )
    parser.add_argument(
        "--img_size", type=int, default=28, help="size of each image dimension"
    )
    parser.add_argument(
        "--channels", type=int, default=1, help="number of image channels"
    )
    parser.add_argument(
        "--sample_interval", type=int, default=400, help="interval betwen image samples"
    )
    parser.add_argument("--c1", type=str, default="LUAD", help="cancer type for c1")
    parser.add_argument("--c2", type=str, default="STAD", help="cancer type for c2")
    parser.add_argument(
        "--save_path", type=str, default="models/g", help="model save path"
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=100000000,
        help="top n genes to calculate correlation",
    )
    opt = parser.parse_args()
    opt.c1 = c1
    opt.c2 = c2
    print(opt)

    df1 = pd.read_csv(
        "raw_survival/" + opt.c1 + "_surv.txt_clean", index_col=None, sep="\t"
    )
    patient_data1 = {}
    for row in df1.values:
        patient_data1[row[0]] = [row[1], row[2]]

    df2 = pd.read_csv(
        "raw_survival/" + opt.c2 + "_surv.txt_clean", index_col=None, sep="\t"
    )
    patient_data2 = {}
    for row in df2.values:
        patient_data2[row[0]] = [row[1], row[2]]


    df = pd.read_csv("new_data.csv", index_col=None)
    c1 = df[df["cancer"] == opt.c1]
    c2 = df[df["cancer"] == opt.c2]

    name_mapping = pd.read_csv("name_mapping.txt", sep="\t", dtype=str)
    print(name_mapping.columns)
    id2realname = {}
    for index, row in name_mapping.iterrows():
        name = row["Approved symbol"]
        gene_id = row["NCBI Gene ID"]
        id2realname[gene_id] = name

    print(id2realname)
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

    for index, row in c1.iterrows():
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
    c2_data = []
    c2_count = {}
    time_list2 = []
    surv_list2 = []
    for index, row in c2.iterrows():
        patient = row["id"]
        if patient in patient_data2:
            time_list2.append(patient_data2[patient][0])
            surv_list2.append(patient_data2[patient][1])
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

    from numpy.random import shuffle

    def batches(entries):
        shuffle(entries)
        for i in range(0, len(entries), 10):
            yield entries[i : i + 10]

    def batch(iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx : min(ndx + n, l)]

    # Loss function
    adversarial_loss = torch.nn.BCELoss()

    # Initialize generator and discriminator
    generator = Generator(num_genes)
    discriminator = Discriminator(num_genes)

    cuda = True if torch.cuda.is_available() else False
    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(
        generator.parameters(), lr=1 * opt.lr, betas=(opt.b1, opt.b2)
    )
    optimizer_D = torch.optim.Adam(
        discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)
    )

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # ----------
    #  Training
    # ----------
    d_list = []
    g_list = []
    # transformer = Normalizer().fit(kirc_late_data)
    # kirc_late_data = transformer.transform(kirc_late_data)
    # transformer = Normalizer().fit(stad_late_data)
    # stad_late_data = transformer.transform(stad_late_data)
    co = []
    p = []
    scores = []
    match = []
    val = []
    co3 = 0
    co4 = 0
    p1 = [[], [], [], [], []]
    p2 = [[], [], [], [], []]
    small1 = [[], [], [], [], []]
    small2 = [[], [], [], [], []]
    min_c = [
        0,
        0,
        0,
        0,
        0,
    ]
    select_p = [0, 0, 0, 0, 0]
    best_labels = [[], [], [], [], []]
    sil_score = [0, 0, 0, 0, 0]
    from sklearn.decomposition import PCA

    for epoch in range(opt.n_epochs):
        print(epoch)
        data_c1 = torch.cuda.FloatTensor(c1_data)

        generator.eval()
        with torch.no_grad():
            gen_gene = generator(data_c1)
            gen_gene = gen_gene.cpu().detach().numpy()
            """
            c2_count_generated = {}
            for i, data in enumerate(gen_gene):
                for (gene, score) in enumerate(data):
                    gene = index_2_name[gene]
                    if gene in c2_count_generated:
                        c2_count_generated[gene] += score
                    else:
                        c2_count_generated[gene] = score
            sorted_count_c2 = sorted(
                c2_count.items(), key=lambda kv: kv[1], reverse=True
            )
            sorted_count_c2_generated = sorted(
                c2_count_generated.items(), key=lambda kv: kv[1], reverse=True
            )

            name_2_index = {}
            count = 0
            for i, (a, b) in enumerate(sorted_count_c2):
                if i < opt.top_n:
                    name_2_index[a] = count
                    count += 1
            colis = [0] * count
            colis_gen = [0] * count

            for (a, b) in sorted_count_c2:
                if a in name_2_index:
                    index = name_2_index[a]
                    colis[index] = b
            for (a, b) in sorted_count_c2_generated:
                if a in name_2_index:
                    index = name_2_index[a]
                    colis_gen[index] = b
            """
            from scipy import stats

            # print(stats.spearmanr(colis, colis_gen, axis=0))
            # co.append(stats.spearmanr(colis, colis_gen, axis=0)[0])
            c1_new = np.copy(c1_data)
            c2_new = np.copy(c2_data)
            c1_new = np.array(c1_new, dtype=float)
            c2_new = np.array(c2_new, dtype=float)
            from sklearn.decomposition import PCA
            from sklearn.metrics.pairwise import euclidean_distances

            # print(np.max(gen_gene, axis=1))
            # print(np.min(gen_gene, axis=1))
            # print(gen_gene[0])
            # gen_gene = np.where(gen_gene > 0.5, 1, 0)
            c1_new = PCA(n_components=32, random_state=2).fit_transform(c1_new)
            c2_new = np.array(gen_gene, dtype=float)
            c2_new = PCA(n_components=32, random_state=2).fit_transform(c2_new)
            dis_mat1 = euclidean_distances(c1_new, c1_new)
            dis_mat2 = euclidean_distances(c2_new, c2_new)
            # print("aaaaaaa", original_gen_gene[:10], "aaa", original_c2[:10])
            from sklearn.metrics import silhouette_score

            label = np.zeros(len(gen_gene) + len(c2_data))
            label[len(gen_gene) :] = 1
            score = silhouette_score(np.concatenate([gen_gene, c2_data]), label)
            print(score)
            scores.append(score)
            from numpy import diff

            for n in range(2, 7):
                cluster1 = KMeans(n_clusters=n, random_state=3).fit(dis_mat1)
                cluster2 = KMeans(n_clusters=n, random_state=3).fit(dis_mat2)
                import collections

                # print(cluster2.labels_)
                # print(collections.Counter(cluster2.labels_))
                results1 = multivariate_logrank_test(
                    time_list1, cluster1.labels_, event_observed=surv_list1
                )
                results2 = multivariate_logrank_test(
                    time_list1,
                    cluster2.labels_[: len(time_list1)],
                    event_observed=surv_list1,
                )
                print(results1.p_value, results2.p_value)
                p1[n - 2].append(results1.p_value)
                p2[n - 2].append(results2.p_value)
                min_value = min(collections.Counter(cluster1.labels_).values())
                print(min_value)
                small1[n - 2].append(min_value)
                min_value = min(collections.Counter(cluster2.labels_).values())
                small2[n - 2].append(min_value)
                if min_value > min_c[n - 2]:
                    min_c[n - 2] = min_value
                    select_p[n - 2] = results2.p_value
                    sil_score[n - 2] = score
                    best_labels[n - 2] = cluster2.labels_

            if epoch > 8:
                dy = diff(scores) / 1
                # dy = diff(dy) / 1

            if epoch % 10 == 9:
                import matplotlib.pyplot as plt

                plt.clf()
                plt.plot(scores)
                plt.savefig("plots/" + opt.c1 + "/" + opt.c2 + "/" + "scores")
                plt.clf()
                plt.plot(dy)
                plt.savefig("plots/" + opt.c1 + "/" + opt.c2 + "/" + "dy")
                for i in range(2, 7):
                    plt.clf()
                    plt.plot(p1[i - 2])
                    plt.plot(p2[i - 2])
                    # plt.plot(co)
                    plt.legend(
                        ["clustering with original c1", "clustering with c2_generated"]
                    )
                    plt.title("c1 = " + opt.c1 + ",c2 = " + opt.c2)
                    plt.yscale("log")
                    plt.xlabel("epochs")
                    plt.ylabel("p_value")
                    plt.show()

                    plt.savefig(
                        "plots/" + opt.c1 + "/" + opt.c2 + "/" + str(i) + "_pvalue"
                    )

                    plt.clf()
                    plt.plot(small1[i - 2])
                    plt.plot(small2[i - 2])
                    # plt.plot(co)
                    plt.legend(
                        ["clustering with original c1", "clustering with c2_generated"]
                    )
                    plt.title("c1 = " + opt.c1 + ",c2 = " + opt.c2)
                    plt.xlabel("epochs")
                    plt.ylabel("smallest number of patients in cluster")
                    plt.show()

                    plt.savefig(
                        "plots/"
                        + opt.c1
                        + "/"
                        + opt.c2
                        + "/"
                        + str(i)
                        + "_smallest_num_patients"
                    )

        batches_c1 = batches(np.copy(c1_data))

        generator.train()
        for i, data_c1 in enumerate(batches_c1):
            # Adversarial ground truths
            data_c1 = torch.cuda.FloatTensor(data_c1)
            shuffle(c2_data)
            data_c2 = c2_data[: data_c1.shape[0]]
            data_c2 = torch.cuda.FloatTensor(data_c2)
            valid = Variable(Tensor(data_c1.size(0), 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(data_c1.size(0), 1).fill_(0.0), requires_grad=False)
            # Configure input
            real_gene = Variable(data_c2.type(Tensor))
            # Generate a batch of images
            gen_gene = generator(data_c1)
            # print(torch.sum(gen_gene, axis=1))

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(real_gene), valid)
            fake_loss = adversarial_loss(discriminator(gen_gene.detach()), fake)

            if i % 2 == 0:
                d_loss = real_loss
                pred = (discriminator(real_gene) > 0.5).double()
                num_correct = sum((pred == valid).double())
            else:
                d_loss = fake_loss
                pred = (discriminator(gen_gene.detach()) > 0.5).double()
                num_correct = sum((pred == fake).double())
            # d_loss = (real_loss + fake_loss) / 2
            pred = (discriminator(real_gene) > 0.5).double()
            num_correct1 = sum((pred == valid).double())
            pred = (discriminator(gen_gene.detach()) > 0.5).double()
            num_correct2 = sum((pred == fake).double())
            # print("d", num_correct1, num_correct2)
            # if num_correct1 < valid.shape[0] / 2 or num_correct2 < valid.shape[0] / 2:
            d_loss.backward()
            optimizer_D.step()

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_gene), valid)
            pred = (discriminator(gen_gene.detach()) > 0.5).double()
            num_correct = sum((pred == valid).double())
            # print("g", num_correct)
            g_loss.backward()
            optimizer_G.step()

    append_list_as_row(
        "results.csv",
        [
            opt.c1,
            opt.c2,
            p1[0][0],
            select_p[0],
            p1[1][0],
            select_p[1],
            p1[2][0],
            select_p[2],
            p1[3][0],
            select_p[3],
            p1[4][0],
            select_p[4],
        ],
    )
    with open("plots/" + opt.c1 + "/" + opt.c2 + "/labels", "w") as text_file:
        for n in range(2, 7):
            text_file.write(",".join([str(i) for i in best_labels[n - 2]]) + "\n")
    with open("plots/" + opt.c1 + "/" + opt.c2 + "/scores", "w") as text_file:
        for n in range(2, 7):
            text_file.write(str(sil_score[n - 2]) + "\n")
    # torch.save(generator.state_dict(), opt.save_path)
    import matplotlib.pyplot as plt

    plt.clf()
    plt.plot(p1)
    plt.plot(p2)
    # plt.plot(co)
    plt.legend(["clustering with original c1", "clustering with c2_generated"])
    plt.title("c1 = " + opt.c1 + "c2 = " + opt.c2)
    plt.yscale("log")
    plt.xlabel("epochs")
    plt.ylabel("p_value")
    # plt.show()


if __name__ == "__main__":
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

    for c1 in cancer_types:
        os.mkdir("plots/" + c1)
        for c2 in cancer_types:
            if c1 != c2:
                os.mkdir("plots/" + c1 + "/" + c2)
                main(c1, c2)

