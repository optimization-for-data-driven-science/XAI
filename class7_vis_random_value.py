from __future__ import print_function
from __future__ import division
from builtins import range
from builtins import int
from builtins import dict

from PIL import Image


import argparse
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable as V
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torch.backends.cudnn as cudnn

import torch.nn.functional as F

import torchvision
import torchvision.datasets as dset
import torchvision.transforms as T

from model import ConvNet
from model2 import FullyConnectedNet

import numpy as np

import matplotlib.pyplot as plt

from dataprocess2 import *

import os
import copy

def save_image(X, filename):
    X = torchvision.utils.make_grid(X[:16], nrow=4)
    # torchvision.utils.save_image(grid, "ORG.png")
    X = np.transpose(X.numpy(), (1, 2, 0))
    X = (X * 255).astype(np.ubyte)
    im = Image.fromarray(X)
    (width, height) = (im.width * 4, im.height * 4)
    im_resized = im.resize((width, height), resample=None)
    im_resized.save(filename)

    return im_resized

def main():
    
    MNIST_train = FM(7)
    loader_train = DataLoader(MNIST_train, batch_size=64, shuffle=False)
    dtype = torch.FloatTensor

    fname = "MNIST.pth"


    num_correct = 0
    num_samples = 0


    weights = pickle.load(open(fname, "rb"))
    C1 = weights[0]
    BC1 = weights[1]
    C2 = weights[2]
    BC2 = weights[3]
    F1 = weights[4]
    BF1 = weights[5]
    F2 = weights[6]
    BF2 = weights[7]

    loss_f = nn.CrossEntropyLoss()


    ig = pickle.load(open("ig_single_class_7_all_dim_random.pkl", "rb"))
    ig2 = ig[0].clone()
    
    a, b, c, d = 18, 0, 27, 27
    ig = ig[0]
    ig = ig[:, :, :, a: c + 1, b: d + 1]
    print(ig.shape)
    ig = ig.sum(-1).sum(-1).sum(-1).sum(-1)
    ig2 = ig2.sum(-1).sum(-1).sum(-1).sum(-1)
    print(ig.shape)

    print("=" * 20)
    for i in range(64):
    	print(ig[i].item(), ig2[i].item())
    print("=" * 20)

    threshold = 0

    mask = torch.ones((1, 64))

    w = []
    for i in range(64):

        w.append(((ig[i].item()), i))

    w.sort(key=lambda x: x[0], reverse=True)

    # for i in range(64):
    #     if ig[i] > threshold:
    #         mask[0][i] = 0

    for j in range(48):
    	mask[0][w[j][1]] = 0

    print(mask.sum())

    loader_train = DataLoader(MNIST_train, batch_size=64, shuffle=False)
    for X_, y_ in loader_train:

        N = len(X_) 

        X = V(X_.type(dtype), requires_grad=False)
        y = V(y_.type(dtype), requires_grad=False).long()

        IG = torch.zeros_like(X_)
        
        for k in range(8):

            steps = 240
            X_base = torch.rand(X.shape)
            delta = (X - X_base) / steps
            grad_acc = torch.zeros_like(X)

            for i in range(steps):

                X_p = (i + 1) * delta + X_base

                X_p.requires_grad = True

                tmp = F.conv2d(X_p, C1, BC1)
                tmp = F.max_pool2d(F.relu(tmp), 2)
                tmp = F.conv2d(tmp, C2, BC2)
                tmp = F.max_pool2d(F.relu(tmp), 2)
                tmp = tmp.view(-1, 160)
                tmp = F.linear(tmp, F1, BF1)
                tmp = F.relu(tmp)
                tmp = mask.repeat(N, 1) * tmp
                logits = F.linear(tmp, F2, BF2)
                probs = F.softmax(logits, dim=1)
                probs = probs[torch.arange(N), y]
                loss = probs.sum()
                loss.backward()
                # print(X_p.grad)
                grad_acc += X_p.grad
                X_p.grad.zero_()

            IG += grad_acc * delta

        IG /= 8

        print(IG.shape)
        # exit(0)
        IG_in = IG.clone()[:, :, a: c + 1, b: d + 1]
        IGI = IG_in.sum()
        IGO = IG.sum() - IGI

        print(IGI)
        print(IGO)
                


        



        break



if __name__ == '__main__':
    main()