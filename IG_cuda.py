from __future__ import print_function
from __future__ import division
from builtins import range
from builtins import int
from builtins import dict

from tqdm import tqdm

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


def main(args):
    
    loader_train, loader_test = loadData(args)
    device = torch.device("cuda")
    
    # weights = train(args, loader_train, loader_test, dtype)

    fname = "MNIST.pth"
    # pickle.dump(weights, open(, "wb"))

    # model = torch.load(fname)

    test2(fname, loader_train, device)

    print("Training done, model save to %s :)" % fname)



def test2(fname, loader_train, device):


    def partial_forward(X):

        return F.conv2d(X, C1, BC1)

    num_correct = 0
    num_samples = 0
    # model.eval()

    weights = pickle.load(open(fname, "rb"))
    C1 = weights[0].to(device)
    BC1 = weights[1].to(device)
    C2 = weights[2].to(device)
    BC2 = weights[3].to(device)
    F1 = weights[4].to(device)
    BF1 = weights[5].to(device)
    F2 = weights[6].to(device)
    BF2 = weights[7].to(device)

    loss_f = nn.CrossEntropyLoss()

    nnn = 0

    for X, y in loader_train:

        X = X.to(device)
        y = y.to(device)


        N = len(X)

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

        # print(probs1)
        # print(probs2)
        print(probs1 - probs2)

        steps = 240
        delta = (X - X_zero) / steps
        secondterm = []
        firstterm = []
        kk = 0

        
        # for j in range(20):
        #   for k in range(24):
        #       for l in range(24):
        #           for i in range(steps):

        #               X_p = (i + 1) * delta + X_zero

        #               X_p.requires_grad = True

        #               tmp = F.conv2d(X_p, C1, BC1)
        #               # tmp_ = tmp.clone().detach()
        #               tmp_ = tmp.clone().detach()
        #               tmp[0][j][k][j].backward()
        #               secondterm.append(X_p.grad.clone())
        #               X_p.grad.zero_()
        #               tmp_.requires_grad = True
        #               tmp = F.max_pool2d(F.relu(tmp_), 2)
        #               tmp = F.conv2d(tmp, C2, BC2)
        #               tmp = F.max_pool2d(F.relu(tmp), 2)
        #               tmp = tmp.view(-1, 800)
        #               tmp = F.linear(tmp, F1, BF1)
        #               tmp = F.relu(tmp)
        #               preds = F.linear(tmp, F2, BF2)

        #               loss = loss_f(preds, y)
        #               # print(loss)

        #               grad_ = torch.autograd.grad(loss, tmp_)[0]
        #               grad = grad_[0][j][k][l]

        #               # print((grad_ == 0).sum())
        #               # print(torch.argmax(grad_))
        #               # print(torch.max(grad_))

        #               firstterm.append(grad.clone())

        #           acc = torch.zeros_like(secondterm[0])

        #           for f, s in zip(firstterm, secondterm):
        #               acc += f * s

        #           acc = (acc * delta).sum() / steps

        #           print(j, k, l, kk, acc.item())
        #           kk += 1



        # # print(firstterm)
        # # print(secondterm)

        # exit(0)

        # for j in range(20):
        #   for k in range(24):
        #       for l in range(24):

        total = torch.zeros((4, 5, 24, 24, 4, 1, 28, 28), device=device)
        total_n = torch.zeros((1, 5, 24, 24), device=device)

        for i in tqdm(range(steps)):

            X_p = (i + 1) * delta + X_zero
            X_p2 = X_p.clone().detach()

            X_p.requires_grad = False
            X_p2.requires_grad = True

            # torch.cuda.empty_cache()

            # tmp = F.conv2d(X_p, C1, BC1)
            # tmp_ = tmp.clone().detach()



            secondterm = torch.autograd.functional.jacobian(partial_forward, X_p2, vectorize=True)

            # torch.cuda.empty_cache()


            tmp_ = F.conv2d(X_p, C1, BC1).clone().detach()
            tmp_.requires_grad = True
            tmp = F.max_pool2d(F.relu(tmp_), 2)
            tmp = F.conv2d(tmp, C2, BC2)
            tmp = F.max_pool2d(F.relu(tmp), 2)
            tmp = tmp.view(-1, 160)
            tmp = F.linear(tmp, F1, BF1)
            tmp = F.relu(tmp)
            logits = F.linear(tmp, F2, BF2)
            probs = F.softmax(logits, dim=1)
            probs = probs[torch.arange(N), y]
            loss = probs.sum()

            firstterm = torch.autograd.grad(loss, tmp_, only_inputs=True)[0]
            # print(firstterm.shape)
            # print(secondterm.shape)

            prod = firstterm.view(4, 5, 24, 24, 1, 1, 1, 1) * secondterm
            total += prod

    


        total = total * delta.view(1, 1, 1, 1, 4, 1, 28, 28)

        print(total.sum(-1).sum(-1).sum(-1).sum(-1).sum(-1).sum(-1).sum(-1))
        

        nnn += 1 
        total_n += total.sum(-1).sum(-1).sum(-1).sum(-1).mean(0)

        print(total_n.shape)
        print()

        ig_now = total_n / nnn
        pickle.dump(ig_now.clone().detach().cpu(), open("ig.pkl", "wb"))
        # exit(0)

            


        # grad_acc /= steps
        # IG = grad_acc * delta
        # IG = IG.abs()
        # IG = (IG - torch.min(IG)) / (torch.max(IG) - torch.min(IG))

        # print(torch.max(IG))

        # print(IG)




        # grid = torchvision.utils.make_grid(IG[:64], nrow=8)
        # torchvision.utils.save_image(grid, "IG.png")

        # grid = torchvision.utils.make_grid(X[:64], nrow=8)
        # torchvision.utils.save_image(grid, "ORG.png")
        # break

        # with torch.no_grad():

        #   tmp = F.conv2d(X, C1, BC1)
        #   tmp = F.max_pool2d(F.relu(tmp), 2)
        #   tmp = F.conv2d(tmp, C2, BC2)
        #   tmp = F.max_pool2d(F.relu(tmp), 2)
        #   tmp = tmp.view(-1, 800)
        #   tmp = F.linear(tmp, F1, BF1)
        #   tmp = F.relu(tmp)
        #   logits = F.linear(tmp, F2, BF2)

        # _, preds = logits.max(1)

        # num_correct += (preds == y).sum()
        # num_samples += preds.size(0)



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
            tmp = tmp.view(-1, 800)
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


    C1 = torch.empty(20, 1, 5, 5)
    BC1 = torch.empty(20)
    C2 = torch.empty(50, 20, 5, 5)
    BC2 = torch.empty(50)
    F1 = torch.empty(500, 800)
    BF1 = torch.empty(500)
    F2 = torch.empty(10, 500)
    BF2 = torch.empty(10)

    weights = [C1, BC1, C2, BC2, F1, BF1, F2, BF2]
    for weight in weights:
        torch.nn.init.normal_(weight, std=5e-4)
        weight.requires_grad = True
        
    loss_f = nn.CrossEntropyLoss()


    num_epochs = 10
    learning_rate = 1e-2

        
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
            tmp = tmp.view(-1, 800)
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

