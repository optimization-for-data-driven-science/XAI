from __future__ import print_function
from __future__ import division
from builtins import range
from builtins import int
from builtins import dict


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

from dataprocess import *

import os
import copy


def main(args):
    
    loader_train, loader_test, loader_train2, loader_test2  = loadData(args)
    dtype = torch.FloatTensor
    
    # weights = train(args, loader_train, loader_test, dtype)

    fname = "MNIST.pth"
    # pickle.dump(weights, open(fname, "wb"))

    # model = torch.load(fname)

    test2(fname, loader_test, dtype)

    # print("Training done, model save to %s :)" % fname)



def test2(fname, loader_test, dtype):
    num_correct = 0
    num_samples = 0
    # model.eval()

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

    for X_, y_ in loader_test:


        X_, y_ = pickle.load(open("X.pkl", "rb"))

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




        steps = 240
        delta = (X - X_zero) / steps
        grad_acc = torch.zeros_like(X)
        for i in range(steps):

            X_p = (i + 1) * delta + X_zero

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

        

        grid = torchvision.utils.make_grid(X[:25], nrow=5)
        torchvision.utils.save_image(grid, "ORG.png")
        grid = np.transpose(grid.numpy(), (1, 2, 0))
        grid = (grid * 255).astype(np.ubyte)
        from PIL import Image
        im=Image.fromarray(grid)
        (width, height) = (im.width * 4, im.height * 4)
        im_resized = im.resize((width, height),resample=None)
        # im_resized = im.resize((width, height),resample=Image.NEAREST)
        im_resized.save("ORG.png")



        # grad_acc /= steps
        IG = grad_acc * delta
        print(IG.shape)

        IG_vis = IG[0].flatten().tolist()
        IG_vis.sort()
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"]})

        plt.figure(figsize=(3, 2))
        # plt.tight_layout(True)
        plt.plot(IG_vis)
        plt.ylim(-0.06, 0.06)
        plt.savefig("stats_back.eps")
        plt.savefig("stats_back.png")

        plt.show()







        



        

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
            img_integrated_gradient, _, _ = visualize(IG_tmp, img, clip_above_percentile=99, clip_below_percentile=30, overlay=False, channel=[0, 255, 0])
            img_integrated_gradient = np.transpose(img_integrated_gradient, (2, 0, 1))
            IG3[i] = torch.tensor(img_integrated_gradient)
            img_integrated_gradient, _, _ = visualize(-IG_tmp, img, clip_above_percentile=99, clip_below_percentile=70, overlay=False, channel=[255, 0, 0])
            img_integrated_gradient = np.transpose(img_integrated_gradient, (2, 0, 1))
            IG3[i] += torch.tensor(img_integrated_gradient)

        grid = torchvision.utils.make_grid(IG3[:25], nrow=5)
        torchvision.utils.save_image(grid, "IG_new.png")
        grid = np.transpose(grid.numpy(), (1, 2, 0))
        grid = (grid * 255).astype(np.ubyte)
        from PIL import Image
        im=Image.fromarray(grid)
        (width, height) = (im.width * 4, im.height * 4)
        im_resized2 = im.resize((width, height),resample=None)
        # im_resized = im.resize((width, height),resample=Image.NEAREST)
        im_resized2.save("IG_new.png")

        new_img = Image.blend(im_resized, im_resized2, 0.75)
        new_img.save("new.png")


        exit(0)

        print(IG_)
        print((IG_ - GT).abs())

        exit(0)

        print(IG.shape)

        print(torch.max(IG))

        print(IG)




        grid = torchvision.utils.make_grid(IG[:64], nrow=8)
        torchvision.utils.save_image(grid, "IG.png")

        grid = torchvision.utils.make_grid(X[:64], nrow=8)
        torchvision.utils.save_image(grid, "ORG.png")
        break

        with torch.no_grad():

            tmp = F.conv2d(X, C1, BC1)
            tmp = F.max_pool2d(F.relu(tmp), 2)
            tmp = F.conv2d(tmp, C2, BC2)
            tmp = F.max_pool2d(F.relu(tmp), 2)
            tmp = tmp.view(-1, 800)
            tmp = F.linear(tmp, F1, BF1)
            tmp = F.relu(tmp)
            logits = F.linear(tmp, F2, BF2)

        _, preds = logits.max(1)

        num_correct += (preds == y).sum()
        num_samples += preds.size(0)



    # accuracy = float(num_correct) / num_samples * 100
    # print('\nAccuracy(X) = %.2f%%' % accuracy)

def test(weights, loader_test, dtype):
    num_correct = 0
    num_samples = 0
    # model.eval()


    C1 = weights[0]
    BC1 = weights[1]
    C2 = weights[2]
    BC2 = weights[3]
    F1 = weights[4]
    BF1 = weights[5]
    F2 = weights[6]
    BF2 = weights[7]

    for X_, y_ in loader_test:

        X = V(X_.type(dtype), requires_grad=False)
        y = V(y_.type(dtype), requires_grad=False).long()

        with torch.no_grad():

            tmp = F.conv2d(X, C1, BC1)
            tmp = F.max_pool2d(F.relu(tmp), 2)
            tmp = F.conv2d(tmp, C2, BC2)
            tmp = F.max_pool2d(F.relu(tmp), 2)
            tmp = tmp.view(-1, 160)
            tmp = F.linear(tmp, F1, BF1)
            tmp = F.relu(tmp)
            logits = F.linear(tmp, F2, BF2)

        _, preds = logits.max(1)

        num_correct += (preds == y).sum()
        num_samples += preds.size(0)

    accuracy = float(num_correct) / num_samples * 100
    print('\nAccuracy(X) = %.2f%%' % accuracy)

    # model.train()

def train(args, loader_train, loader_test, dtype):

    # model = ConvNet()
    # model = model.type(dtype)
    # model.train()


    C1 = torch.empty(5, 1, 5, 5)
    BC1 = torch.empty(5)
    C2 = torch.empty(10, 5, 5, 5)
    BC2 = torch.empty(10)
    F1 = torch.empty(64, 160)
    BF1 = torch.empty(64)
    F2 = torch.empty(10, 64)
    BF2 = torch.empty(10)

    weights = [C1, BC1, C2, BC2, F1, BF1, F2, BF2]
    for weight in weights:
        if weight.dim() > 1:
            torch.nn.init.xavier_normal_(weight)
        else:
            torch.nn.init.constant_(weight, 1 / weight.numel())
        weight.requires_grad = True
        
    loss_f = nn.CrossEntropyLoss()


    num_epochs = 35
    learning_rate = 1e-3

        
    print('\nTraining %d epochs with learning rate %.4f' % (num_epochs, learning_rate))
    
    optimizer = optim.Adam(weights, lr=learning_rate)
    
    for epoch in range(num_epochs):
        
        print('\nTraining epoch %d / %d ...\n' % (epoch + 1, num_epochs))
        
        for i, (X_, y_) in enumerate(loader_train):

            X = V(X_.type(dtype), requires_grad=False)
            y = V(y_.type(dtype), requires_grad=False).long()

            tmp = F.conv2d(X, C1, BC1)
            tmp = F.max_pool2d(F.relu(tmp), 2)
            tmp = F.conv2d(tmp, C2, BC2)
            tmp = F.max_pool2d(F.relu(tmp), 2)
            tmp = tmp.view(-1, 160)
            tmp = F.linear(tmp, F1, BF1)
            tmp = F.relu(tmp)
            preds = F.linear(tmp, F2, BF2)

            loss = loss_f(preds, y)
            
            if (i + 1) % args.print_every == 0:
                print('Batch %d done, loss = %.7f' % (i + 1, loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Batch %d done, loss = %.7f' % (i + 1, loss.item()))
        
        test(weights, loader_test, dtype)

    return weights

def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='./dataset', type=str,
                        help='path to dataset')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='size of each batch of cifar-10 training images')
    parser.add_argument('--print-every', default=50, type=int,
                        help='number of iterations to wait before printing')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    main(args)

