import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader 
from torch.autograd import Variable
from data_loader import PanoticDataset 


import torch.nn as nn
import torch.nn.functional as F
import torch

from pytictoc import TicToc

t = TicToc()

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()

# dataset configuration
parser.add_argument("--n_joint", type=int, default=18, help="number of joint")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")

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
    def __init__(self, in_channels, out_channels, kernel, stride, pad, ratio):
        super(ModifiedResidualBlock, self).__init__()
        
        self.in_channels=in_channels
        self.out_channels = out_channels
        self.kernel = kernel
        self.pad = pad
        self.ratio = ratio

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=pad)
        self.prelu = nn.PReLU()
        self.ratio = ratio
    
    def downsample(self,x):
        return F.interpolate(x,x.size()[2])
        
    def forward(self, x):        
        out = self.conv(x)
        residual = self.prelu(out) + math.sqrt(1 - self.ratio) * F.interpolate(x,size=out.size()[2])
        out = math.sqrt(self.ratio) * residual + math.sqrt(1 - self.ratio) * x

        return out

class TransposedModifiedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, pad, ratio):
        super(TransposedModifiedResidualBlock, self).__init__()
        self.conv = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size=kernel, stride=stride, padding=pad
        )
        self.prelu = nn.PReLU()
        self.ratio = ratio

    def forward(self, x):
        out = self.conv(x)
        residual = self.prelu(out) + math.sqrt(1 - self.ratio) * x
        out = math.sqrt(self.ratio) * residual + math.sqrt(1 - self.ratio) * x

        return out


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
 
        self.conv_layers = []

        self.encoder, conv1 = self.make_encoder()
        self.decoder, conv2 = self.make_decoder()

        # ! fix later
        self.conv_layers.append([conv1, conv2]) # ! fix later
    
    def make_encoder(self):
        layers = []
        conv_layers = []
        layers_channels = [2048, 384, 384, 512, 512, 768, 1024]

        for i in range(1, len(layers_channels)):
            layer = ModifiedResidualBlock(
                layers_channels[i - 1], layers_channels[i], 4, 2, 1, 1 / i
            )
            layers.append(layer)
            conv_layers.append(layer.conv)

        return nn.Sequential(*layers), conv_layers

    def make_decoder(self):
        layers = []
        conv_layers = []
        layers_channels = [1024, 1024, 768, 512, 512, 512, 6 + 3 * (opt.n_joint - 1)]
        ratio = [
            [1.4, 1.6],
            [2.2, 2.8],
            [3.6, 4.6],
            [5.8, 7.2],
            [8.8, 10.6],
            [12.6, 14.8],
            17.2,
        ]

        for i in range(1, len(layers_channels) - 1):
            block = ModifiedResidualBlock(
                layers_channels[i - 1], layers_channels[i], 3, 1, 1, 1 / ratio[i-1][0]
            )
            tr_block = TransposedModifiedResidualBlock(
                layers_channels[i], layers_channels[i], 4, 2, 1, 1 / ratio[i-1][1]
            )
            layers.append(block)
            layers.append(tr_block)

            conv_layers.append(block.conv)
            conv_layers.append(tr_block.conv)

        layers.append(
            ModifiedResidualBlock(512, 6 + 3 * (opt.n_joint - 1), 4, 2, 1, ratio[-1])
        )  # M-1 = without pelvis
        return nn.Sequential(*layers), conv_layers

    def forward(self, x):
        # input = 3 * M * 2

        out = self.encoder(x)
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
        self.layer7 = nn.Conv1d(1024, 1, kernel_size=1, stride=1, padding=1)

        self.conv_layers.append(self.layer1.conv)
        self.conv_layers.append(self.layer2.conv)
        self.conv_layers.append(self.layer3.conv)
        self.conv_layers.append(self.layer4.conv)
        self.conv_layers.append(self.layer5.conv)
        self.conv_layers.append(self.layer6.conv)
        self.conv_layers.append(self.layer7)

    def forward(self, x):

        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)

        return self.layer8(out)


# !!! Minimizes MSE instead of BCE
adversarial_loss = torch.nn.MSELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Configure data loader
dataset = PanoticDataset(dataset="panoptic", json_path="train_list.json")
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=opt.batch_size, shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, data in enumerate(dataloader):
         
        data = data.cuda()
        # Adversarial ground truths
        # ! in the paper. real set to 1, fake set to -1
        # ! check if / 64 -> make int
        real = Variable(Tensor(data.shape[1] , 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(data.shape[1] , 1).fill_(-1.0), requires_grad=False)
        c_fake = Variable(Tensor(data.shape[1], 1).fill_(0.2361), requires_grad=False)

        # Configure input
        real_imgs = Variable(data.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, data.shape)))

        # Generate a batch of images
        # batch regularization loss
        gen_imgs = generator(z)
        mean, std = generator.conv_layers.mean(), generator.conv_layers.std()

        gen_br_loss = ( 1/ (6 + 3(opt.n_joint))* (adversarial_loss(mean, 0) + adversarial_loss(std, 0)))

        # Loss measures generator's ability to fool the discriminator

        g_loss = (
            1 / (data.shape[0] / 64) * adversarial_loss(discriminator(gen_imgs), c_fake)
            + gen_br_loss
        )

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        mean, std = discriminator.conv_layers.mean(), discriminator.conv_layers.std()
        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        dis_br_loss = (
            1
            / (6 + 3(opt.n_joint))
            * (adversarial_loss(mean, 0) + adversarial_loss(std, 0))
        )
        real_loss = adversarial_loss(discriminator(real_imgs), real)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)

        d_loss = 1 / (data.shape[0] / 64) * (real_loss + fake_loss) + dis_br_loss

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(
                gen_imgs.data[:25],
                "images/%d.png" % batches_done,
                nrow=5,
                normalize=True,
            )
