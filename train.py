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
from dataloader import getTrainingData
import matplotlib.pyplot as plt
from model import *

# random seed setting
random.seed(10)
torch.manual_seed(10)
torch.cuda.manual_seed_all(10)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=60, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=128, help='size of the batches')
parser.add_argument('--lr_d', type=float, default=2e-4, help='initial learning rate of disciminator')
parser.add_argument('--lr_g', type=float, default=1e-3, help='initial learning rate of generator')
parser.add_argument('--output_str', type=str, default='./output', help='dir num of output')
parser.add_argument('--dim_noise', type=int, default=54, help='dimension of noise')
parser.add_argument('--dim_dis_latent', type=int, default=10, help='dimension of discrete latent')
parser.add_argument('--Q_loss_weight', type=float, default=0.25, help='Weight of info loss')

opt = parser.parse_args()
print(opt)

os.makedirs(opt.output_str, exist_ok=True)
os.makedirs(os.path.join(opt.output_str, 'checkpoint/'), exist_ok=True)
os.makedirs(os.path.join(opt.output_str, 'vis/'), exist_ok=True)

dataloader = getTrainingData(opt.batch_size)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# Model Definition
dim_inputs = opt.dim_noise + opt.dim_dis_latent
netG = Generator(in_dim=dim_inputs).to(device)
netD = Discriminator().to(device)
netG.apply(weights_init)
netD.apply(weights_init)


# optimizer
# setup optimizer
optimD = optim.Adam(
    [{'params': netD.main.parameters()}, {'params': netD.D.parameters()}],
    lr=opt.lr_d,
    betas=(0.5, 0.999)
)
optimG = optim.Adam(
    [{'params': netG.parameters()}, {'params': netD.Q.parameters()}],
    lr=opt.lr_g,
    betas=(0.5, 0.999)
)


# to one-hot vector
def to_categorical(y, num_columns):
    """Returns one-hot encoded Variable"""
    y_cat = torch.zeros((y.shape[0], num_columns))
    y_cat[range(y.shape[0]), y] = 1.0
    y_cat = y_cat.view(y.shape[0], num_columns, 1, 1)
    y_cat = y_cat.to(device)

    return y_cat


# validation data, use val data to visualize the generated data at each epoch
fix_noise = torch.randn([10, opt.dim_noise, 1, 1], device=device).unsqueeze(1).repeat(1, 10, 1, 1, 1).view(100, opt.dim_noise, 1, 1)
y = np.array([num for _ in range(10) for num in range(10)])
y_cat = to_categorical(y, opt.dim_dis_latent)
val_data_inputs = torch.cat([fix_noise, y_cat], dim=1)


# Loss for discrimination between real and fake images.
criterionD = nn.BCELoss()
# Loss for discrete latent code.
criterionQ = nn.CrossEntropyLoss()

# they are used to record loss at each epoch so as to plot loss curve
plot_D, plot_G, plot_info = [], [], []

# start training 
for epoch in range(opt.n_epochs):
    netD.train()
    netG.train()
    
    # these are loss for disciminator, generator, and classifier
    # they can help you record the loss in a csv file
    print_D, print_G, print_info = 0, 0, 0

    for i, batch in enumerate(dataloader, 0):
        real_data = batch['x'].to(device)
        batch_size = real_data.shape[0]
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ############################
        
        ############################
        # TO DO HERE
        # 1. OptimD.zero_grad()
	# 2. Input real data to netD to get predition, then calculate the real_loss = -log(D(x)) and backward
	# 3. Sample noise and one-hot vector, and concatenate them as generator_inputs
	# 3. Input generator_inputs to Generator to get fake data, and input fake data to netD to get predition, then calculate the rake_loss = -log(1 - D(G(z))) and backward
        # 4. OptimD.step()
        ############################            

        
        
        
        
        
        ############################
        # (2) Update G & Q network: maximize log(D(G(z))) + LI(G,Q)
        ############################
        
        ############################
        # TO DO HERE
        # 1. OptimG.zero_grad()
	# 2. Input fake data to netD again to get prediction and latent
	# 3. Calculate the G_loss = -log(D(G(z))) and info_loss = max(LI(G,Q)) and backward
        # 4. OptimG.step()
        ############################         
        

        
        
        
        
        
        print_D    += D_loss.item()
        print_G    += reconstruct_loss.item()
        print_info += info_loss.item()

    print_D    /= len(dataloader)
    print_G    /= len(dataloader)
    print_info /= len(dataloader)
    
    plot_D.append(print_D)
    plot_G.append(print_G)
    plot_info.append(print_info)    


    print('[%d/%d] Loss_D: %.4f, Loss_G: %.4f, Loss_Q: %.4f'
            % (epoch, opt.n_epochs,
               print_D, print_G, print_info))

    # this help to save model
    torch.save({
            'netG': netG,
            'netD': netD
        }, os.path.join(opt.output_str, 'checkpoint/model_{}.pt'.format(epoch))
    )

    # this will help you show the result of your generator
    netD.eval()
    netG.eval()
    with torch.no_grad():
        fake = netG(val_data_inputs)
        vutils.save_image(fake.detach(), os.path.join(opt.output_str, 'vis/result_{}.png'.format(epoch)), nrow=10)
    
# plot loss curve    
plt.title('Discriminator loss', fontsize=18)
plt.plot(range(1, opt.n_epochs+1), plot_D, 'red', label='D_loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('D_Loss')
plt.savefig(os.path.join(opt.output_str, 'D_loss.png'))
plt.clf()

plt.title('Generator loss', fontsize=18)
plt.plot(range(1, opt.n_epochs+1), plot_G, 'blue', label='G_loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('G_Loss')
plt.savefig(os.path.join(opt.output_str, 'G_loss.png'))
plt.clf()

plt.title('Discrete Mutual Information loss', fontsize=18)
plt.plot(range(1, opt.n_epochs+1), plot_info, 'green', label='info_loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('info_loss')
plt.savefig(os.path.join(opt.output_str, 'info_loss.png'))
plt.clf()








