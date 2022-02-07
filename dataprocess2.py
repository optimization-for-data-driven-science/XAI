from __future__ import print_function
from __future__ import division
from builtins import range
from builtins import int
from builtins import dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import sampler
import torch.backends.cudnn as cudnn

import torch.nn.functional as F

import torchvision.datasets as dset
import torchvision.transforms as T

# import numpy as np


class FM(Dataset):
    def __init__(self, cat=0):
        MNIST_train = dset.FashionMNIST('./dataset', train=True, transform=T.ToTensor(), download=True)
        self.cat = cat
        self.data = []
        for i in range(10):
            self.data.append([])

        for image, label in MNIST_train:
            self.data[label].append((image, label))

    def __len__(self):
        return len(self.data[self.cat])

    def __getitem__(self, idx):
        return self.data[self.cat][idx]

class ChunkSampler(sampler.Sampler):
    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start
    
    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))
    
    def __len__(self):
        return self.num_samples

def loadData(args):


    # transform_augment_train = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(32, padding=4), T.ToTensor()])
    transform = T.Compose([T.ToTensor()])

    MNIST_train = dset.FashionMNIST('./dataset', train=True, transform=T.ToTensor(), download=True)

    MNIST_test = dset.FashionMNIST('./dataset', train=False, transform=T.ToTensor(), download=True)

  
    loader_train = DataLoader(MNIST_train, batch_size=48, shuffle=True)
    
    loader_test = DataLoader(MNIST_test, batch_size=64)

    return loader_train, loader_test