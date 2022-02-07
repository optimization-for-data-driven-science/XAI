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

    for X_, y_ in loader_train:

        m1 = []
        e1 = []
        m2 = []
        e2 = []

        N = len(X_)

        X = V(X_.type(dtype), requires_grad=False)
        y = V(y_.type(dtype), requires_grad=False).long()

        X_zero = torch.zeros_like(X)
        for weight in weights:
            weight.requires_grad = False


        tmp = F.conv2d(X, C1, BC1)
        tmp = F.max_pool2d(F.relu(tmp), 2)
        tmp = F.conv2d(tmp, C2, BC2)
        tmp = F.max_pool2d(F.relu(tmp), 2)
        tmp = tmp.view(-1, 160)
        tmp = F.linear(tmp, F1, BF1)
        tmp = F.relu(tmp)
        logits = F.linear(tmp, F2, BF2)
        probs = F.softmax(logits, dim=1)
        probs1 = probs[torch.arange(N), y]



        tmp = F.conv2d(X_zero, C1, BC1)
        tmp = F.max_pool2d(F.relu(tmp), 2)
        tmp = F.conv2d(tmp, C2, BC2)
        tmp = F.max_pool2d(F.relu(tmp), 2)
        tmp = tmp.view(-1, 160)
        tmp = F.linear(tmp, F1, BF1)
        tmp = F.relu(tmp)
        logits = F.linear(tmp, F2, BF2)
        probs = F.softmax(logits, dim=1)
        probs2 = probs[torch.arange(N), y]


        GT = (probs1 - probs2)
        print(GT)



        IG = torch.zeros_like(X)

        
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

        
        X_tmp = X[:16]
        X_tmp = X_tmp.repeat(1, 3, 1, 1)
        print(X_tmp.shape)
        a, b, c, d = 18, 0, 27, 27
        for i in range(16):
            
            for yy in range(b, d + 1):
                X_tmp[i][0][a][yy] = 1
                X_tmp[i][1][a][yy] = 1
                X_tmp[i][2][a][yy] = 0

                X_tmp[i][0][c][yy] = 1
                X_tmp[i][1][c][yy] = 1
                X_tmp[i][2][c][yy] = 0

            for xx in range(a, c + 1):
                X_tmp[i][0][xx][b] = 1
                X_tmp[i][1][xx][b] = 1
                X_tmp[i][2][xx][b] = 0

                X_tmp[i][0][xx][d] = 1
                X_tmp[i][1][xx][d] = 1
                X_tmp[i][2][xx][d] = 0


        X_org = save_image(X, "7_org.png")
        X_org_bd = save_image(X_tmp, "7_org_bd.png")

        


        IG_ = IG.clone().view(64, -1).sum(dim=1)

        from visualization import visualize


        IG3 = torch.empty((64, 3, 28, 28))
        for i in range(64):
            img = X[i].clone()
            img = img.repeat(3, 1, 1)
            img = np.transpose(img.numpy(), (1, 2, 0))
            IG_tmp = IG[i].clone()
            IG_tmp = IG_tmp.repeat(3, 1, 1)
            IG_tmp = np.transpose(IG_tmp.numpy(), (1, 2, 0))
            img_integrated_gradient, m1_, e1_ = visualize(IG_tmp, img, clip_above_percentile=99, clip_below_percentile=30, overlay=False, channel=[0, 255, 0])
            img_integrated_gradient = np.transpose(img_integrated_gradient, (2, 0, 1))
            IG3[i] = torch.tensor(img_integrated_gradient)
            img_integrated_gradient, m2_, e2_ = visualize(-IG_tmp, img, clip_above_percentile=99, clip_below_percentile=70, overlay=False, channel=[255, 0, 0])
            img_integrated_gradient = np.transpose(img_integrated_gradient, (2, 0, 1))
            IG3[i] += torch.tensor(img_integrated_gradient)
            m1.append(m1_)
            e1.append(e1_)
            m2.append(m2_)
            e2.append(e2_)

        IG = save_image(IG3, "7_IG.png")
        IG_overlay = Image.blend(X_org, IG, 0.75)
        IG_overlay_bd = Image.blend(X_org_bd, IG, 0.75)
        IG_overlay.save("7_IG_overlay.png")
        IG_overlay_bd.save("7_IG_overlay_bd.png")


        

        print(IG_)
        print((IG_ - GT).abs())



        break

    ig = pickle.load(open("ig_single_class_7_all_dim_random.pkl", "rb"))
    ig2 = ig[0].clone()
    

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

        X_zero = torch.zeros_like(X)
        for weight in weights:
            weight.requires_grad = False


        tmp = F.conv2d(X, C1, BC1)
        tmp = F.max_pool2d(F.relu(tmp), 2)
        tmp = F.conv2d(tmp, C2, BC2)
        tmp = F.max_pool2d(F.relu(tmp), 2)
        tmp = tmp.view(-1, 160)
        tmp = F.linear(tmp, F1, BF1)
        tmp = F.relu(tmp)
        tmp = mask.repeat(N, 1) * tmp
        logits = F.linear(tmp, F2, BF2)
        probs = F.softmax(logits, dim=1)
        probs1 = probs[torch.arange(N), y]



        tmp = F.conv2d(X_zero, C1, BC1)
        tmp = F.max_pool2d(F.relu(tmp), 2)
        tmp = F.conv2d(tmp, C2, BC2)
        tmp = F.max_pool2d(F.relu(tmp), 2)
        tmp = tmp.view(-1, 160)
        tmp = F.linear(tmp, F1, BF1)
        tmp = F.relu(tmp)
        tmp = mask.repeat(N, 1) * tmp
        logits = F.linear(tmp, F2, BF2)
        probs = F.softmax(logits, dim=1)
        probs2 = probs[torch.arange(N), y]


        GT = (probs1 - probs2)
        print(probs1)
        print(probs2)
        print(GT)
        print("*" * 20)




        IG = torch.zeros_like(X)
        
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

        
        X_tmp = X[:16]
        X_tmp = X_tmp.repeat(1, 3, 1, 1)
        print(X_tmp.shape)
        # a, b, c, d = 13, 0, 20, 17
        for i in range(16):
            
            for yy in range(b, d + 1):
                X_tmp[i][0][a][yy] = 1
                X_tmp[i][1][a][yy] = 1
                X_tmp[i][2][a][yy] = 0

                X_tmp[i][0][c][yy] = 1
                X_tmp[i][1][c][yy] = 1
                X_tmp[i][2][c][yy] = 0

            for xx in range(a, c + 1):
                X_tmp[i][0][xx][b] = 1
                X_tmp[i][1][xx][b] = 1
                X_tmp[i][2][xx][b] = 0

                X_tmp[i][0][xx][d] = 1
                X_tmp[i][1][xx][d] = 1
                X_tmp[i][2][xx][d] = 0


        X_org = save_image(X, "7_org_after_zero.png")
        X_org_bd = save_image(X_tmp, "7_org_bd_after_zero.png")

        


        IG_ = IG.clone().view(64, -1).sum(dim=1)

        from visualization import visualize


        IG3 = torch.empty((64, 3, 28, 28))
        for i in range(64):
            img = X[i].clone()
            img = img.repeat(3, 1, 1)
            img = np.transpose(img.numpy(), (1, 2, 0))
            IG_tmp = IG[i].clone()
            IG_tmp = IG_tmp.repeat(3, 1, 1)
            IG_tmp = np.transpose(IG_tmp.numpy(), (1, 2, 0))
            img_integrated_gradient, _, _ = visualize(IG_tmp, img, clip_above_percentile=99, clip_below_percentile=30, overlay=False, channel=[0, 255, 0], force_m_e=True, m=m1[i], e=e1[i])
            img_integrated_gradient = np.transpose(img_integrated_gradient, (2, 0, 1))
            IG3[i] = torch.tensor(img_integrated_gradient)
            img_integrated_gradient, _, _ = visualize(-IG_tmp, img, clip_above_percentile=99, clip_below_percentile=70, overlay=False, channel=[255, 0, 0], force_m_e=True, m=m2[i], e=e2[i])
            img_integrated_gradient = np.transpose(img_integrated_gradient, (2, 0, 1))
            IG3[i] += torch.tensor(img_integrated_gradient)

        IG = save_image(IG3, "7_IG_after_zero.png")
        IG_overlay = Image.blend(X_org, IG, 0.75)
        IG_overlay_bd = Image.blend(X_org_bd, IG, 0.75)
        IG_overlay.save("7_IG_overlay_after_zero.png")
        IG_overlay_bd.save("7_IG_overlay_bd_after_zero.png")


        

        print(IG_)
        print((IG_ - GT).abs())

        # print((m1, e1))
        # print((m2, e2))
        # print((m3, e3))
        # print((m4, e4))



        break



if __name__ == '__main__':
    main()