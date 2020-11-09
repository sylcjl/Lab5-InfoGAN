#!/usr/bin/env python
# -*- coding: utf-8 -*-
import struct
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class mnist_dataset(Dataset):
    def __init__(self):
        with open('./mnist/train-images.idx3-ubyte', 'rb') as f:
            zero, data_type, dims = struct.unpack('>HBB', f.read(4))
            shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
            self.x = np.fromstring(f.read(), dtype=np.uint8).reshape(shape)
            self.x = np.expand_dims(self.x, axis=1)
            
    def __getitem__(self, idx):
        x = transforms.Compose([transforms.ToTensor()])(self.x[idx]).permute(1, 2, 0)
        sample = {'x': x}

        return sample

    def __len__(self):
        return self.x.shape[0]


def mnist_dataloader(batch_size=128):
    from torchvision import datasets
    dataset = datasets.MNIST('./dataset', transform=transforms.ToTensor(), download=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    return dataloader


def getTrainingData(batch_size=128):
    training_data = mnist_dataset()
    dataloader_training = DataLoader(training_data, batch_size,
                                     shuffle=True, num_workers=0, pin_memory=False)
    return dataloader_training


if __name__ == "__main__":
    data_loader = getTrainingData(32)
    pass