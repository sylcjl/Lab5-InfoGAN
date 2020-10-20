import struct
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

class mnist_dataset(Dataset):
    def __init__(self):
        with open('./mnist/train-images-idx3-ubyte', 'rb') as f:
            zero, data_type, dims = struct.unpack('>HBB', f.read(4))
            shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
            self.x = np.fromstring(f.read(), dtype=np.uint8).reshape(shape)
            self.x = np.expand_dims(self.x, axis = 1)
            
    def __getitem__(self, idx):
        x = torch.from_numpy(np.float32(self.x[idx]))

        sample = {'x': x}


        return sample

    def __len__(self):
        return self.x.shape[0]
     

def getTrainingData(batch_size=128):
    training_data = mnist_dataset()
    dataloader_training = DataLoader(training_data, batch_size,
                                     shuffle=True, num_workers=10, pin_memory=False)
    return dataloader_training


   
    