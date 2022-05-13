#!/usr/bin/python3

import argparse
import sys
import os

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
#from torch.autograd import Variable
import torch

from models import Generator
from datasets import ImageDataset

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='datasets/horse2zebra/', help='root directory of the dataset')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--size', type=int, default=128, help='size of the data (squared assumed)')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--generator_A2B', type=str, default='output_1/19_netG_A2B.pth', help='A2B generator checkpoint file')
opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# Networks
netG_A2B = Generator(opt.input_nc, opt.output_nc)


if opt.cuda:
    netG_A2B.cuda() #netG_A2B.to(torch.device('cuda')) new



#print(torch.load(opt.generator_A2B))

# Load state dicts
if opt.cuda:
    netG_A2B.load_state_dict({k.replace('module.',''):v for k,v in torch.load(opt.generator_A2B).items()})

else:
    netG_A2B.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(opt.generator_A2B, map_location=torch.device('cpu')).items()})


# Set model's test mode e.x.  set dropout and batch normalization layers to evaluation mode
netG_A2B.eval()


# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)

# Dataset loader
transforms_ = [ transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)) ]
dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, mode='test'), 
                        batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)
###################################

###### Testing######

# Create output dirs if they don't exist
if not os.path.exists('output/A'):
    os.makedirs('output/A')
if not os.path.exists('output/B'):
    os.makedirs('output/B')


#############################################
img = torch.rand(1, 1,128,128).cpu()
model1 = netG_A2B.cpu()
output_trace=netG_A2B(img)
traced_script_module = torch.jit.trace(model1, img)
print("jit save")
traced_script_module.save('./aifont.pt')
print("jit end")

###############################################
