import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torchsummary import summary

from torch.utils.data import DataLoader 
from torch.autograd import Variable
from data_loader import PanopticDataset 
from utils import utils


import torch.nn as nn
import torch.nn.functional as F
import torch
import random
import matplotlib.pyplot as plt


from pytictoc import TicToc

random.seed(42)

os.makedirs("images", exist_ok=True)
parser = argparse.ArgumentParser()

# dataset configuration
parser.add_argument("--n_joint", type=int, default = 19, help="number of joint")
parser.add_argument("--n_epochs", type=int, default = 50, help="number of epochs of training") # ! need to fix
parser.add_argument("--batch_size", type=int, default = 4, help="size of the batches") # ! need to fix
parser.add_argument("--input_frame",type=int, default = 2 ** 10, help = "size of the input frame")
parser.add_argument("--dis_input_frame",type=int, default = 120, help = "size of the discriminator input frame")
parser.add_argument("--skip_frame", type=int, default=30, help="skip frames at the discriminator input")
parser.add_argument("--sample_interval",type=int, default=10)

# training configuration 
parser.add_argument("--lr", type=float, default=0.00001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5,help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2",type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu",type=int, default=8,help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
opt = parser.parse_args()

cuda = True if torch.cuda.is_available() else False

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 5.0)  # * mod
        torch.nn.init.constant_(m.bias.data, 0.0)

class ModifiedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, padding, ratio):
        super(ModifiedResidualBlock, self).__init__()
         
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding)
        self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding)
        self.conv2.weight.data.fill_(1)
        self.conv2.weight.detach()

        self.bn = nn.BatchNorm1d(out_channels)
        self.prelu = nn.PReLU()
        self.ratio1 = math.sqrt(ratio)
        self.ratio2 = math.sqrt(1-ratio)
        
    def forward(self, input):
        
        x = self.conv1(input)
        x = self.bn(x)
        x = self.prelu(x[0].clone())

        downsample = self.conv2(input)
        
        return self.ratio1 * x + self.ratio2 * downsample
     
# @todo list
# def affine_transform()
# a learned affine transformation, which multiplies
# each feature elementwise by a learned scale-factor and adds to each
# feature elementwise a learned per-feature bias.


class TransposedModifiedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride,padding,ratio):
        super(TransposedModifiedResidualBlock, self).__init__()
        self.tr_conv1 = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=kernel, stride=stride,padding=padding)
        self.tr_conv2 = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=kernel, stride=stride,padding=padding)
        self.tr_conv2.weight.data.fill_(1)
        self.tr_conv2.weight.detach()
        
        self.bn = nn.BatchNorm1d(out_channels) 
        self.prelu = nn.PReLU()
        self.ratio1 = math.sqrt(ratio)
        self.ratio2 = math.sqrt(1-ratio)
    
    def forward(self, input):
        x = self.tr_conv1(input)
        x = self.bn(x)
        
        x = self.prelu(x[0].clone())

        upsample = self.tr_conv2(input) 

        return self.ratio1 * x + self.ratio2 * upsample


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
 
        self.conv_layers = []

        self.encoder = self.make_encoder()
        self.decoder  = self.make_decoder()
  
    def make_encoder(self):
        layers = [] 
        layers_channels = [opt.n_joint * 4, 384, 384, 512, 512, 768, 1024]

        for i in range(1, len(layers_channels)): 
            layer = ModifiedResidualBlock(
                layers_channels[i - 1], layers_channels[i], 4, 2, 1, 1 / i
            )
            layers.append(layer) 

        return nn.Sequential(*layers) 
    
    def make_decoder(self):
        layers = []
        conv_layers = []
        layers_channels = [1024,1024,768,768, 512, 512, 512, 3 * (opt.n_joint - 1)]
        ratio = [
            [1.4, 1.6],
            [2.2, 2.8],
            [3.6, 4.6],
            [5.8, 7.2],
            [8.8, 10.6],
            [12.6, 14.8],
            17.2,
        ]
        
        block = ModifiedResidualBlock(layers_channels[0], layers_channels[0], 1, 1, 0, 1 / 1)
        layers.append(block)
        
        for i in range(1, len(layers_channels) - 1):
            block = ModifiedResidualBlock(
                layers_channels[i - 1], layers_channels[i], 3, 1, 1, 1 / ratio[i-1][0]
            )
            tr_block = TransposedModifiedResidualBlock(
                layers_channels[i], layers_channels[i], 4, 2, 1, 1 / ratio[i-1][1]
            ) 
            layers.append(block)
            layers.append(tr_block)

        layers.append(
            ModifiedResidualBlock(512, layers_channels[-1], 3, 1, 1, 1 / ratio[-1])
        )  # M-1 = without pelvis

        return nn.Sequential(*layers)
 
    def forward(self, input): 
        
        x = self.encoder(input)
        out = self.decoder(x)

        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv_layers = []
        # ! n of JOINT 
        self.layer1 = ModifiedResidualBlock(3 * (opt.n_joint - 1), 512, 4, 2, 1, 1 / 1)
        self.layer2 = ModifiedResidualBlock(512, 512, 4, 2, 1, 1 / 2)
        self.layer3 = ModifiedResidualBlock(512, 512, 4, 2, 1, 1 / 3)
        self.layer4 = ModifiedResidualBlock(512, 512, 4, 2, 1, 1 / 4)
        self.layer5 = ModifiedResidualBlock(512, 1024, 4, 2,1, 1 / 5)
        self.layer6 = ModifiedResidualBlock(1024, 1024, 4, 2, 1, 1 / 6) 
        self.adv_layer = nn.Linear(1024 * 3, 1) # 32768
            
    def forward(self, x):
        
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer5(out)
        out = self.layer6(out) 
        out = out.view(out.clone().shape[0], -1)
        
        return self.adv_layer(out)


# !!! Minimizes MSE instead of BCE
adversarial_loss = torch.nn.MSELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()


if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    
summary(generator,(19*4,1024)) # 1024 frames as input
# summary(discriminator,(18*3,120)) # 120 frames -> discriminate real/fake

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Configure data loader
dataset = PanopticDataset(dataset='my_panoptic',json_path='test_list.json',input_frame = opt.input_frame)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True) 

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

discr_loss = []
gen_loss = []
iter = 0
for epoch in range(opt.n_epochs):
    
    for i, (mask_data, data) in enumerate(dataloader):
        
        mask_data = mask_data.to(device='cuda').permute((0,2,1)) # [4,1024,76] (19*4)
        data = data.to(device='cuda').permute((0,2,1)) # [4,54,1024] (18*3)
        
        # Adversarial ground truths
        # ! in the paper. real set to 1, fake set to -1
        # ! check if / 64 -> make int
        
        # Configure input
        # (batch_size, frames, length)
        # discriminator input
        real_imgs = Variable(data.type(Tensor)) # discriminator input
        
        # discriminator output
        real = Variable(Tensor(data.shape[0], 1).fill_(1.0), requires_grad=False)  
        fake = Variable(Tensor(data.shape[0], 1).fill_(-1.0), requires_grad=False) 
        c_fake = Variable(Tensor(data.shape[0], 1).fill_(0.2361), requires_grad=False)  
        
        # -----------------
        #  Train Generator/ train with fake data
        # ----------------- 
        optimizer_G.zero_grad()
 
        # Generate a batch of images        
        gen_imgs = generator(mask_data.float())
        
        # forward
        g_selected_fr = random.randint(0,100)
        dis =  discriminator(gen_imgs[:,:,g_selected_fr: g_selected_fr + opt.dis_input_frame])
        g_loss = torch.mean((dis -c_fake) ** 2)
        
        # backward
        optimizer_G.zero_grad()
        g_loss.backward() 
        
        
        # gradient descent 
        optimizer_G.step()
 
        
        if i % 6  == 0: 
            # ---------------------
            #  Train Discriminator
            # ---------------------  
            optimizer_D.zero_grad()
            d_selected_fr = random.randint(0,100) 
            
            # forward
            real_loss =  torch.mean((discriminator(real_imgs[:,:,g_selected_fr: g_selected_fr + opt.dis_input_frame]) - real) ** 2) 
            fake_loss =  torch.mean((discriminator(gen_imgs[:,:,g_selected_fr: g_selected_fr + opt.dis_input_frame].detach()) - fake) ** 2)
            d_loss = 0.5 * (real_loss+ fake_loss)
            
            # backward 
            d_loss.backward()
            
            # gradient descent
            optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch + 1, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )
        
        discr_loss.append(d_loss.item())
        gen_loss.append(g_loss.item())
        
        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            
            print(gen_imgs[0])
            print(real_imgs[0])
            # utils()
            
    
plt.plot(discr_loss, color = 'red')
plt.plot(gen_loss, color = 'blue')
plt.legend(['discriminator','generator'])

plt.savefig("training loss.png")
plt.show()

    