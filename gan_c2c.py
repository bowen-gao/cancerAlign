import argparse
import os
import numpy as np
import math
import pandas as pd

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
from sklearn.preprocessing import Normalizer

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
            *block(num_genes, 16, normalize=False),
            # *block(512, 512),
            # *block(512, 512),
            nn.Linear(16, num_genes),
            nn.Sigmoid()
        )

    def forward(self, z):
        out = self.model(z)
        return out


class Discriminator(nn.Module):
    def __init__(self, num_genes):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(num_genes, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def forward(self, gene_exp):
        validity = self.model(gene_exp)

        return validity


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
        default=0.5,
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
    parser.add_argument("--c1", type=str, default="KIRC", help="cancer type for c1")
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

    """
    load data
    KIRC and STAD
    """

    df = pd.read_csv("data.csv", index_col=None)
    c1 = df[df["cancer"] == opt.c1]
    c2 = df[df["cancer"] == opt.c2]
    c1_late = c1[c1["stage"] == "late"]
    c1_early = c1[c1["stage"] == "early"]
    c2_late = c2[c2["stage"] == "late"]
    c2_early = c2[c2["stage"] == "early"]

    print(c1_early, c1_late, c2_early, c2_late)

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

    c2_count_early = {}
    c2_count_late = {}
    c1_count_late = {}
    c1_count_early = {}

    c1_late_data = []

    for index, row in c1_late.iterrows():
        patient = row["id"]
        genes = row["mut_genes"]
        genes = genes.split(";")
        vec = np.zeros(num_genes)
        for gene in genes:
            if gene in c1_count_late:
                c1_count_late[gene] += 1
            else:
                c1_count_late[gene] = 1
            vec[name_2_index[gene]] = 1
        c1_late_data.append(vec)
    c2_late_data = []
    for index, row in c2_late.iterrows():
        patient = row["id"]
        genes = row["mut_genes"]
        genes = genes.split(";")
        vec = np.zeros(num_genes)
        for gene in genes:
            if gene in c2_count_late:
                c2_count_late[gene] += 1
            else:
                c2_count_late[gene] = 1
            vec[name_2_index[gene]] = 1
        c2_late_data.append(vec)
    c1_early_data = []
    for index, row in c1_early.iterrows():
        patient = row["id"]
        genes = row["mut_genes"]
        genes = genes.split(";")
        vec = np.zeros(num_genes)
        for gene in genes:
            if gene in c1_count_early:
                c1_count_early[gene] += 1
            else:
                c1_count_early[gene] = 1
            vec[name_2_index[gene]] = 1
        c1_early_data.append(vec)
    c2_early_data = []
    for index, row in c2_early.iterrows():
        patient = row["id"]
        genes = row["mut_genes"]
        genes = genes.split(";")
        vec = np.zeros(num_genes)
        for gene in genes:
            if gene in c2_count_early:
                c2_count_early[gene] += 1
            else:
                c2_count_early[gene] = 1
            vec[name_2_index[gene]] = 1
        c2_early_data.append(vec)

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
    match = []
    val = []
    co3 = 0
    co4 = 0
    for epoch in range(opt.n_epochs):
        print(epoch)
        batches_c1 = batches(c1_late_data[:])
        batches_c2 = batches(c2_late_data[:100])

        for i, data_c1 in enumerate(batches_c1):
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
        data_c1 = torch.cuda.FloatTensor(c1_early_data)
        generator.eval()
        gen_gene = generator(data_c1)
        gen_gene = gen_gene.cpu().detach().numpy()
        c2_count_generated_early = {}
        for i, data in enumerate(gen_gene):
            for (gene, score) in enumerate(data):
                gene = index_2_name[gene]
                if gene in c2_count_generated_early:
                    c2_count_generated_early[gene] += score
                else:
                    c2_count_generated_early[gene] = score
        sorted_count_c2_early = sorted(
            c2_count_early.items(), key=lambda kv: kv[1], reverse=True
        )
        sorted_count_c2_early_generated = sorted(
            c2_count_generated_early.items(), key=lambda kv: kv[1], reverse=True
        )
        sorted_count_c2_late = sorted(
            c2_count_late.items(), key=lambda kv: kv[1], reverse=True
        )
        sorted_count_c1_early = sorted(
            c1_count_early.items(), key=lambda kv: kv[1], reverse=True
        )
        sorted_count_c1_late = sorted(
            c1_count_late.items(), key=lambda kv: kv[1], reverse=True
        )

        name_2_index = {}
        count = 0
        for i, (a, b) in enumerate(sorted_count_c2_early):
            if i < opt.top_n:
                name_2_index[a] = count
                count += 1
        colis = [0] * count
        colis2 = [0] * count
        colis3 = [0] * count
        colis4 = [0] * count
        colis5 = [0] * count
        for (a, b) in sorted_count_c2_early:
            if a in name_2_index:
                index = name_2_index[a]
                colis[index] = b
        for (a, b) in sorted_count_c2_early_generated:
            if a in name_2_index:
                index = name_2_index[a]
                colis2[index] = b
        for (a, b) in sorted_count_c2_late:
            if a in name_2_index:
                index = name_2_index[a]
                colis3[index] = b
        for (a, b) in sorted_count_c1_early:
            if a in name_2_index:
                index = name_2_index[a]
                colis4[index] = b
        for (a, b) in sorted_count_c1_late:
            if a in name_2_index:
                index = name_2_index[a]
                colis5[index] = b
        colis6 = [colis3[i] - colis5[i] + colis4[i] for i in range(len(colis3))]
        from scipy import stats

        print(
            stats.spearmanr(colis, colis2, axis=0),
            len(list(set(colis[:100]) & set(colis2[:100]))),
        )
        print(
            stats.spearmanr(colis, colis3, axis=0),
            len(list(set(colis[:100]) & set(colis3[:100]))),
        )
        print(
            stats.spearmanr(colis, colis4, axis=0),
            len(list(set(colis[:100]) & set(colis4[:100]))),
        )
        print(
            stats.spearmanr(colis, colis6, axis=0),
            len(list(set(colis[:100]) & set(colis4[:100]))),
        )
        co.append(stats.spearmanr(colis, colis2, axis=0)[0])
        p.append(stats.spearmanr(colis, colis2, axis=0)[1])
        match.append(len(list(set(colis[:100]) & set(colis2[:100]))))
        co3 = stats.spearmanr(colis, colis3)[0]
        co4 = stats.spearmanr(colis, colis4)[0]

        data_c1 = torch.cuda.FloatTensor(c1_late_data)
        generator.eval()
        gen_gene = generator(data_c1)
        gen_gene = gen_gene.cpu().detach().numpy()
        c2_count_generated_late = {}
        for i, data in enumerate(gen_gene):
            for (gene, score) in enumerate(data):
                gene = index_2_name[gene]
                if gene in c2_count_generated_late:
                    c2_count_generated_late[gene] += score
                else:
                    c2_count_generated_late[gene] = score
        sorted_count_c2_late_generated = sorted(
            c2_count_generated_late.items(), key=lambda kv: kv[1], reverse=True
        )
        sorted_count_c2_late = sorted(
            c2_count_late.items(), key=lambda kv: kv[1], reverse=True
        )
        name_2_index = {}
        count = 0
        for i, (a, b) in enumerate(sorted_count_c2_late):
            if i < opt.top_n:
                name_2_index[a] = count
                count += 1
        colis_gt = [0] * count
        colis_gen = [0] * count
        for (a, b) in sorted_count_c2_late_generated:
            if a in name_2_index:
                index = name_2_index[a]
                colis_gen[index] = b
        for (a, b) in sorted_count_c2_late:
            if a in name_2_index:
                index = name_2_index[a]
                colis_gt[index] = b
        print(stats.spearmanr(colis_gt, colis_gen, axis=0))
        val.append(stats.spearmanr(colis_gt, colis_gen, axis=0)[0])
    torch.save(generator.state_dict(), opt.save_path)
    import matplotlib.pyplot as plt

    plt.plot(g_list)
    plt.plot(d_list)
    plt.legend(["g", "d"])

    plt.show()
    plt.clf()
    plt.plot(co)
    plt.plot(val)
    print(co, co3, co4)
    plt.plot([co3] * len(co))
    plt.plot([co4] * len(co))
    plt.legend(["generated_c2_early", "validation", "gt_c2_late", "gt_c1_early"])
    plt.title("correlation coefficient for all genes")

    plt.show()
    plt.savefig("c2c")
    """
    plt.plot(match)
    plt.plot([59] * len(match))
    plt.plot([34] * len(match))
    plt.legend(["generated_c2_early", "gt_c2_late", "gt_c1_early"])
    plt.title("number of matched genes in all genes")
    plt.show()
    """


if __name__ == "__main__":
    main()
