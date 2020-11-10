#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torchvision.utils as vutils
from dataloader import getTrainingData, mnist_dataloader
import matplotlib.pyplot as plt
from model import *

# random seed setting
random.seed(142)
torch.manual_seed(142)
torch.cuda.manual_seed_all(142)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=128, help='size of the batches')
parser.add_argument('--lr_d', type=float, default=2e-4, help='initial learning rate of discriminator')
parser.add_argument('--lr_g', type=float, default=1e-3, help='initial learning rate of generator')
parser.add_argument('--output_str', type=str, default='./output', help='dir num of output')
parser.add_argument('--dim_noise', type=int, default=62, help='dimension of noise')
parser.add_argument('--dim_dis_latent', type=int, default=10, help='dimension of discrete latent')
parser.add_argument('--dim_con_latent', type=int, default=2, help='dimension of continuous latent')
parser.add_argument('--Q_loss_weight', type=float, default=1, help='Weight of info loss')
parser.add_argument('--Q_cont_loss_weight', type=float, default=0.1, help='Weight of continuous info loss')

opt = parser.parse_args()
print(opt)


chpt_path = os.path.join(opt.output_str, 'checkpoint2')
vis_path = os.path.join(opt.output_str, 'vis2')

os.makedirs(opt.output_str, exist_ok=True)
os.makedirs(chpt_path, exist_ok=True)
os.makedirs(vis_path, exist_ok=True)

# dataloader = mnist_dataloader(opt.batch_size)
dataloader = getTrainingData(opt.batch_size)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class NormalNLLLoss(nn.Module):
    """
    Calculate the negative log likelihood of normal distribution.
    This needs to be minimised.
    Treating Q(cj | x) as a factored Gaussian.
    """
    def forward(self, x, mu, var):
        logli = -0.5 * (var.mul(2 * np.pi) + 1e-6).log() - (x - mu).pow(2).div(var.mul(2.0) + 1e-6)
        nll = -(logli.sum(1).mean())

        return nll


# to one-hot vector
def to_categorical(y, num_columns):
    """Returns one-hot encoded Variable"""
    y_cat = torch.zeros((y.shape[0], num_columns))
    y_cat[range(y.shape[0]), y] = 1.0
    y_cat = y_cat.view(y.shape[0], num_columns, 1, 1)
    y_cat = y_cat.to(device)

    return y_cat


def sample_noise(batch_size, noise_size, c_disc_size, c_cont_size):
    noise = torch.randn(batch_size, noise_size, 1, 1, device=device)
    c_disc_cat = np.random.randint(0, c_disc_size, size=batch_size)
    c_disc = to_categorical(c_disc_cat, 10)
    c_cont = torch.FloatTensor(batch_size, c_cont_size, 1, 1).uniform_(-1, 1).to(device)
    noise_input = torch.cat((noise, c_disc, c_cont), dim=1)
    return noise_input, c_disc_cat, c_cont


# Model Definition
dim_inputs = opt.dim_noise + opt.dim_dis_latent + opt.dim_con_latent

netG = Generator(in_dim=dim_inputs).to(device)
netShared = SharedNet().to(device)
netD = Discriminator().to(device)
netQ = QNet().to(device)

netG.apply(weights_init)
netShared.apply(weights_init)
netD.apply(weights_init)
netQ.apply(weights_init)

print("=========== NetG ===========")
print(netG.eval())
print("=========== NetShared ===========")
print(netShared.eval())
print("=========== NetD ===========")
print(netD.eval())
print("=========== netQ ===========")
print(netQ.eval())


# optimizer
# setup optimizer
optimD = optim.Adam(
    [{'params': netShared.parameters()}, {'params': netD.parameters()}],
    lr=opt.lr_d,
    betas=(0.5, 0.999)
)

optimG = optim.Adam(
    [{'params': netG.parameters()}, {'params': netQ.parameters()}],
    lr=opt.lr_g,
    betas=(0.5, 0.999)
)

############################
# validation data, use val data to visualize the generated data at each epoch
############################
fix_noise = torch.randn([10, opt.dim_noise, 1, 1], device=device) \
                 .unsqueeze(1).repeat(1, 10, 1, 1, 1).view(100, opt.dim_noise, 1, 1)
# fix_noise = torch.randn(100, opt.dim_noise, 1, 1, device=device)
y = np.array([num for _ in range(10) for num in range(10)])
y_dis = to_categorical(y, opt.dim_dis_latent)
y_con = torch.FloatTensor(100, opt.dim_con_latent, 1, 1).uniform_(-1, 1).to(device)
val_data_inputs = torch.cat([fix_noise, y_dis, y_con], dim=1)

############################
# Loss for discrimination between real and fake images.
# Loss for discrete / continuous latent code.
############################
criterionD = nn.BCELoss().to(device)
criterionQ_disc = nn.CrossEntropyLoss().to(device)
criterionQ_cont = NormalNLLLoss().to(device)

# they are used to record loss at each epoch so as to plot loss curve
plot_D, plot_G, plot_info, plot_info_cont = [], [], [], []

# start training
for epoch in range(opt.n_epochs):
    netD.train()
    netShared.train()
    netG.train()
    netQ.train()

    # these are loss for disciminator, generator, and classifier
    # they can help you record the loss in a csv file
    print_D, print_G, print_info, print_info_cont = 0, 0, 0, 0

    for i, batch in enumerate(dataloader, 0):
        # data_real, _ = batch
        # data_real = data_real.to(device)
        data_real = batch['x'].to(device)
        batch_size = data_real.shape[0]

        ############################
        # GAN training hacks
        # 1. https://github.com/soumith/ganhacks
        # 2. https://machinelearningmastery.com/how-to-code-generative-adversarial-network-hacks/
        ############################

        # # Normalize the input images between -1 and 1
        # data_real += torch.randn_like(data_real) / 100
        # # Tanh as the last layer of the generator output
        # data_real = ((data_real - data_real.min()) / (data_real.max() - data_real.min())) * 2 - 1  # Scale to -1 to 1

        # Noisy Labels
        label_real = torch.full((batch_size, ), 1., device=device, requires_grad=False)
        # label_real += torch.tensor(np.random.uniform(-0.3, 0.2, size=b_size), device=device)
        label_fake = torch.full((batch_size, ), 0., device=device, requires_grad=False)
        # label_fake += torch.tensor(np.random.uniform(0, 0.3, size=b_size), device=device)

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ############################

        optimD.zero_grad()
        shared_real = netShared(data_real)
        output_real = netD(shared_real)
        loss_real = criterionD(output_real, label_real)
        loss_real.backward()

        noise_input, c_disc_idx, c_cont = sample_noise(batch_size, opt.dim_noise,
                                                       opt.dim_dis_latent, opt.dim_con_latent)
        data_fake = netG(noise_input)
        shared_fake = netShared(data_fake.detach())
        output_fake = netD(shared_fake)
        loss_fake = criterionD(output_fake, label_fake)
        loss_fake.backward()

        D_loss = loss_real + loss_fake
        optimD.step()

        ############################
        # (2) Update G & Q network: maximize log(D(G(z))) + LI(G,Q)
        ############################

        optimG.zero_grad()

        shared_gen = netShared(data_fake)
        output_gen = netD(shared_gen)

        # Reconstruct Loss
        reconstruct_loss = criterionD(output_gen, label_real)

        # Information loss
        q_logits, q_mu, q_var = netQ(shared_gen)
        ground_truth_class = torch.tensor(c_disc_idx, device=device, dtype=torch.long)
        loss_disc = criterionQ_disc(q_logits, ground_truth_class) * opt.Q_loss_weight
        loss_cont = criterionQ_cont(c_cont, q_mu, q_var) * opt.Q_cont_loss_weight

        # Total G Loss
        G_loss = reconstruct_loss + loss_disc + loss_cont
        G_loss.backward()
        optimG.step()

        print_D    += D_loss.item()
        print_G    += G_loss.item()
        print_info += loss_disc.item()
        print_info_cont += loss_cont.item()

    print_D    /= len(dataloader)
    print_G    /= len(dataloader)
    print_info /= len(dataloader)
    print_info_cont /= len(dataloader)

    plot_D.append(print_D)
    plot_G.append(print_G)
    plot_info.append(print_info)
    plot_info_cont.append(print_info_cont)

    print('[%d/%d] Loss_D: %.4f, Loss_G: %.4f, Loss_Q: %.4f, Loss_Q_cont: %.4f'
            % (epoch, opt.n_epochs,
               print_D, print_G, print_info, print_info_cont))

    # this help to save model
    torch.save({
            'netG': netG,
            'netD': netD,
            'netShared': netShared,
            'netQ': netQ,
        }, os.path.join(chpt_path, 'model_{}.pt'.format(epoch))
    )

    # this will help you show the result of your generator
    netD.eval()
    netG.eval()
    with torch.no_grad():
        fake = netG(val_data_inputs)
        vutils.save_image(fake.detach(), os.path.join(vis_path, 'result_{}.png'.format(epoch)), nrow=10)

    # plot loss curve
    plt.title('Discriminator loss', fontsize=18)
    plt.plot(range(0, epoch+1), plot_D, 'red', label='D_loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('D_Loss')
    plt.savefig(os.path.join(vis_path, 'D_loss_{}.png'.format(epoch)))
    plt.clf()

    plt.title('Generator loss', fontsize=18)
    plt.plot(range(0, epoch+1), plot_G, 'blue', label='G_loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('G_Loss')
    plt.savefig(os.path.join(vis_path, 'G_loss_{}.png'.format(epoch)))
    plt.clf()

    plt.title('Discrete Mutual Information loss', fontsize=18)
    plt.plot(range(0, epoch+1), plot_info, 'green', label='info_loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('info_loss')
    plt.savefig(os.path.join(vis_path, 'info_loss_{}.png'.format(epoch)))
    plt.clf()

    plt.title('Continuous Mutual Information loss', fontsize=18)
    plt.plot(range(0, epoch+1), plot_info_cont, 'orange', label='info_cont_loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('info_cont_loss')
    plt.savefig(os.path.join(vis_path, 'info_cont_loss_{}.png'.format(epoch)))
    plt.clf()

# https://github.com/LJSthu/info-GAN/blob/master/train.py
# https://medium.com/@falconives/day-57-infogan-b9c46ed51e1e
# https://github.com/ozanciga/gans-with-pytorch/blob/master/infogan/infogan.py
# https://www.jianshu.com/p/fa892c81df60
# https://zhuanlan.zhihu.com/p/73324607
# https://colab.research.google.com/github/MicPie/DepthFirstLearning/blob/master/InfoGAN/DCGAN_MNIST_v5.ipynb
# https://github.com/znxlwm/pytorch-generative-model-collections/blob/master/infoGAN.py
# https://github.com/taeoh-kim/Pytorch_InfoGAN/blob/master/infogan.py


