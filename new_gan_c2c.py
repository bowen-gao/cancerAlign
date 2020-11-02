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

"""
This code is adapted from the PyTorch GAN implemetations:
https://github.com/eriklindernoren/PyTorch-GAN
"""

"""
KIRC late -> STAD late (train)
KIRC early -> STAD early (evaluation)
"""


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
            *block(32, 16, normalize=False),
            # *block(512, 512),
            # *block(512, 512),
            nn.Linear(16, 32),
            # nn.Tanh()
        )

    def forward(self, z):
        out = self.model(z)
        return out


class Discriminator(nn.Module):
    def __init__(self, num_genes):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(32, 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(16, 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, gene_exp):
        validity = self.model(gene_exp)

        return validity


from sklearn.cluster import KMeans


def get_cluster(X, n):
    kmeans = KMeans(n_clusters=n, random_state=0).fit(X)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_epochs", type=int, default=200, help="number of epochs of training"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="size of the batches"
    )
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
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

    """
    load data
    KIRC and STAD
    """

    df = pd.read_csv("data.csv", index_col=None)
    c1 = df[df["cancer"] == opt.c1]
    c2 = df[df["cancer"] == opt.c2]

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
    # lg = torch.optim.lr_scheduler.StepLR(optimizer_G, 5, gamma=0.1, last_epoch=-1)
    # ld = torch.optim.lr_scheduler.StepLR(optimizer_D, 5, gamma=0.1, last_epoch=-1)

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
    match = []
    val = []
    co3 = 0
    co4 = 0
    p1 = []
    p2 = []
    from sklearn.decomposition import PCA

    pca_c1 = PCA(n_components=32, random_state=0)
    pca_c2 = PCA(n_components=32, random_state=0)
    original_c1 = np.copy(c1_data)
    original_c2 = np.copy(c2_data)
    c1_data = pca_c1.fit_transform(c1_data)
    c2_data = pca_c2.fit_transform(c2_data)

    for epoch in range(opt.n_epochs):
        print(epoch)
        batches_c1 = batches(np.copy(c1_data))

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

            """
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, 10, d_loss.item(), g_loss.item())
            )
            """
            batches_done = epoch * 10 + i
            """
            if batches_done % opt.sample_interval == 0:
                save_image(gen_gene.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
            """
        """
        epoch_correct = 0
        epoch_g_loss = 0
        epoch_d_loss = 0
        batches_c1 = batches(c1_late_data[:])
        batches_c2 = batches(c2_late_data[:100])
        for i, (data_c1, data_c2) in enumerate(zip(batches_c1, batches_c2)):
            # Adversarial ground truths
            data_c1 = torch.cuda.FloatTensor(data_c1)
            shuffle(c2_late_data)
            data_c2 = c2_late_data[: data_c1.shape[0]]
            data_c2 = torch.cuda.FloatTensor(data_c2)
            valid = Variable(Tensor(data_c1.size(0), 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(data_c1.size(0), 1).fill_(0.0), requires_grad=False)
            # Configure input
            real_gene = Variable(data_c2.type(Tensor))
            # Generate a batch of images
            gen_gene = generator(data_c1)

            # Loss measures generator's ability to fool the discriminator
            epoch_g_loss += adversarial_loss(discriminator(gen_gene), valid)
            real_loss = adversarial_loss(discriminator(real_gene), valid)
            fake_loss = adversarial_loss(discriminator(gen_gene.detach()), fake)

            epoch_d_loss += (real_loss + fake_loss) / 2
            pred = (discriminator(gen_gene.detach()) > 0.5).double()
            epoch_correct += sum((pred == valid).double())
        print(
            "epoch",
            epoch,
            epoch_correct.item(),
            epoch_d_loss.item() / 10,
            epoch_g_loss.item() / 10,
        )
        
        g_list.append(epoch_g_loss.item() / 10)
        d_list.append(epoch_d_loss.item() / 10)
        if (epoch - 1) % 1000 == 0:
            torch.save(generator.state_dict(), opt.save_path)
        """
        data_c1 = torch.cuda.FloatTensor(c1_data)
        generator.eval()
        gen_gene = generator(data_c1)
        gen_gene = gen_gene.cpu().detach().numpy()
        original_gen_gene = pca_c2.inverse_transform(gen_gene)
        # print(np.max(original_gen_gene, axis=1))
        # print(np.min(original_gen_gene, axis=1))
        c2_count_generated = {}
        for i, data in enumerate(original_gen_gene):
            for (gene, score) in enumerate(data):
                gene = index_2_name[gene]
                if gene in c2_count_generated:
                    c2_count_generated[gene] += score
                else:
                    c2_count_generated[gene] = score

        sorted_count_c2 = sorted(c2_count.items(), key=lambda kv: kv[1], reverse=True)
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

        from scipy import stats

        print(stats.spearmanr(colis, colis_gen, axis=0))
        co.append(stats.spearmanr(colis, colis_gen, axis=0)[0])

        c1_new = np.copy(c1_data)
        c2_new = np.copy(c2_data)
        c1_new = np.array(c1_new, dtype=float)
        c2_new = np.array(c2_new, dtype=float)
        from sklearn.decomposition import PCA
        from sklearn.metrics.pairwise import euclidean_distances

        # pca = PCA(n_components=32)
        # c1_new = PCA(n_components=32, random_state=0).fit_transform(c1_new)
        c2_new = np.array(gen_gene, dtype=float)
        c2_new = np.concatenate([gen_gene, c2_data])
        # c2_new = np.concatenate([original_gen_gene, original_c2])
        # c2_new = pca_c2.inverse_transform(c2_new)
        # c2_new = PCA(n_components=32).fit_transform(c2_new)
        # print(gen_gene[0])
        # print(c2_data[0])
        dis_mat1 = euclidean_distances(c1_new, c1_new)
        dis_mat2 = euclidean_distances(c2_new, c2_new)
        # print("aaaaaaa", original_gen_gene[:10], "aaa", original_c2[:10])
        cluster1 = KMeans(n_clusters=2, random_state=0).fit(dis_mat1)
        cluster2 = KMeans(n_clusters=2, random_state=0).fit(dis_mat2)
        import collections

        # print(cluster2.labels_)
        print(
            collections.Counter(cluster1.labels_),
            collections.Counter(cluster2.labels_[: len(time_list1)]),
            collections.Counter(cluster2.labels_[len(time_list1) :]),
        )
        results1 = multivariate_logrank_test(
            time_list1, cluster1.labels_, event_observed=surv_list1
        )
        results2 = multivariate_logrank_test(
            time_list1, cluster2.labels_[: len(time_list1)], event_observed=surv_list1
        )
        print(results1.p_value, results2.p_value)
        p1.append(results1.p_value)
        p2.append(results2.p_value)
        # print(cluster2.labels_[: len(time_list1)])
        if epoch % 10 == 0:
            import matplotlib.pyplot as plt

            plt.clf()
            plt.plot(p1)
            plt.plot(p2)
            plt.legend(["p1", "p2"])
            plt.title("p")

            plt.show()
            plt.savefig("c2c_p2_4")
        # ld.step()
        # lg.step()
    torch.save(generator.state_dict(), opt.save_path)
    import matplotlib.pyplot as plt

    plt.clf()
    plt.plot(p1)
    plt.plot(p2)
    plt.legend(["p1", "p2"])
    plt.title("p")

    plt.show()
    plt.savefig("c2c_p_4")


if __name__ == "__main__":
    main()
