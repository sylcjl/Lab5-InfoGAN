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
            #################
            #  To Do
            #################
        )
        
    def forward(self, x):
        output = self.main(x)
        return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        # shared layer of discriminator
        self.main = nn.Sequential(
            #################
            #  To Do
            #################
        )
        
        # Dsicriminator branch
        self.D = nn.Sequential(
            #################
            #  To Do
            #################
        )
        
        # Info branch
        self.Q = nn.Sequential(
            #################
            #  To Do
            #################
        )
        

    def forward(self, x):
        h = self.main(x)
        real_or_fake = self.D(h).view(-1,1)
        info = self.Q(h).squeeze()
        
        return real_or_fake, info
