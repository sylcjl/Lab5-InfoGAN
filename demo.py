#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
from time import time
import torchvision.utils as vutils

seed = 10
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


dim_noise = 62
dim_dis_latent = 10
dim_con_latent = 2
CHECKPOINT_DIR = os.path.join("test", "chackpoints")
CHECKPOINT_NAME = "model_193.pt"
CHECKPOINT_FP = os.path.join(CHECKPOINT_DIR, CHECKPOINT_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_categorical(y, num_columns):
    """Returns one-hot encoded Variable"""
    y_cat = torch.zeros((y.shape[0], num_columns))
    y_cat[range(y.shape[0]), y] = 1.0
    y_cat = y_cat.view(y.shape[0], num_columns, 1, 1)
    y_cat = y_cat.to(device)

    return y_cat


fix_noise = torch.randn([10, dim_noise, 1, 1], device=device) \
                 .unsqueeze(1).repeat(1, 10, 1, 1, 1).view(100, dim_noise, 1, 1)
# fix_noise = torch.randn(100, opt.dim_noise, 1, 1, device=device)
y = np.array([num for _ in range(dim_dis_latent) for num in range(dim_dis_latent)])
y_dis = to_categorical(y, dim_dis_latent)
y_con = torch.FloatTensor(100, dim_con_latent, 1, 1).uniform_(-1, 1).to(device)
val_data_inputs = torch.cat([fix_noise, y_dis, y_con], dim=1)


checkpoint = torch.load(CHECKPOINT_FP)
netG = checkpoint["netG"]
netG.eval()

fake = netG(val_data_inputs)
result_path = os.path.join('demo_{}.png'.format(int(time())))
vutils.save_image(fake.detach(), result_path, nrow=10)

print("Result generated. Please check {}".format(result_path))
print("Process ends.")
