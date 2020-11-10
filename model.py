#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from torch import nn


'''
 TO DO: define your Generator and Discriminator here
'''
class Generator(nn.Module):
    def __init__(self, in_dim=64):
        super(Generator, self).__init__()
        
        # Generator 
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_dim, 512, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),

            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2, bias=False),

            # nn.Tanh(),  # Tanh as the last layer of the generator output
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        output = self.main(x)
        return output


class SharedNet(nn.Module):
    def __init__(self):
        super(SharedNet, self).__init__()

        # shared layer of discriminator
        self.main = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),

            nn.Conv2d(128, 256, kernel_size=7, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

    def forward(self, x):
        out = self.main(x)
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # Discriminator branch
        self.main = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.main(x)
        return out.squeeze()


class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()

        # Info branch
        self.main = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )
        self.Q_disc = nn.Sequential(
            nn.Conv2d(128, 10, kernel_size=(1, 1)),
        )
        self.Q_cont_mu = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=(1, 1))
        )
        self.Q_cont_var = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=(1, 1))
        )

    def forward(self, x):
        out = self.main(x)
        disc_logits = self.Q_disc(out).squeeze()
        mu = self.Q_cont_mu(out).squeeze()
        var = self.Q_cont_var(out).squeeze().exp()
        return disc_logits, mu, var
