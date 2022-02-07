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



from dataprocess3 import *

import os
import copy
import random
# import pyperclip

# import math

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def main(args):
    
    loader_train, loader_test = loadData(args)
    device = torch.device("cuda")
    
    # weights = train(args, loader_train, loader_test, dtype)

    fname = "MNIST.pth"
    # pickle.dump(weights, open(, "wb"))

    # model = torch.load(fname)

    f_res = open("cat0_keep_part_1.csv", "w")


    for m in range(17):
        rnd, ig, rnd_vec, ig_vec = test2(fname, loader_test, device, m, num_run=50)
        print(rnd)
        print(ig)
        print(rnd_vec)
        print(ig_vec)

        f_res.write("%.4f,"%rnd)
        for num in rnd_vec:
            f_res.write("%.4f,"%num)
        f_res.write("\n")
        f_res.write("%.4f,"%ig)
        for num in ig_vec:
            f_res.write("%.4f,"%num)
        f_res.write("\n")

    f_res.close()



    # s = ""
    # N = 5
    # pp = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1]
    # for p in pp:
    #     rnd_s = 0 
    #     ig_s = 0 
    #     rnd_s_vec = np.zeros(10)
    #     ig_s_vec = np.zeros(10)
    #     for i in range(N):
    #         rnd, ig, rnd_vec, ig_vec = test2(fname, loader_test, device, p)
    #         rnd_s += rnd 
    #         ig_s += ig
    #         rnd_s_vec += rnd_vec
    #         ig_s_vec += ig_vec
    #     print(p, rnd_s / N, ig_s / N)
    #     print(rnd_s_vec / N)
    #     print(ig_s_vec / N)
    #     s += ("%.2f\t%.2f\t" % (p, rnd_s / N))
    #     s += "\t"
    #     for x in rnd_s_vec:
    #         s += "%.2f\t" % (x / N * 100) 
    #     s += "\n\t%.2f\t\t" % (ig_s / N)
    #     # print(s)
    #     for x in ig_s_vec:
    #         s += "%.2f\t" % (x / N * 100)
    #     s += "\n"

    #     pyperclip.copy(s)


    print("Training done, model save to %s :)" % fname)



def test2(fname, loader_test, device, m, num_run):


    import matplotlib.pyplot as plt


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

    ig = pickle.load(open("ig_single_class0_new_45.pkl", "rb"))

    # # print(ig.shape)
    # ig2 = ig.view(-1).clone().tolist()
    # ig2.sort()
    # plt.plot(ig2)
    # plt.savefig("cat7.png")
    # # plt.show()
    # # exit(0)


    w = []
    for i in range(64):

        w.append(((ig[0][i].item()), i))



    w.sort(key=lambda x: x[0], reverse=False)

    total_acc = 0 
    total_acc_vec = np.zeros(10)

    for counter in range(num_run):
        mask = torch.zeros((1, 64)) + 1
        mask = mask.to(device)

        idx = torch.from_numpy(np.random.choice(np.arange(64),m,replace=False)).long()
        print(idx)
        print(len(idx))
        if len(idx) != 0:
            mask[:, idx] = 0

        acc_per_class_N = np.zeros(10)
        acc_per_class_N_correct = np.zeros(10)


        for X, y in loader_test:

            N, _, _, _ = X.shape


            with torch.no_grad():

                X = X.to(device)
                tmp = F.conv2d(X, C1, BC1)
                tmp = F.max_pool2d(F.relu(tmp), 2)
                tmp = F.conv2d(tmp, C2, BC2)
                tmp = F.max_pool2d(F.relu(tmp), 2)
                tmp = tmp.view(-1, 160)
                tmp = F.linear(tmp, F1, BF1)
                tmp = F.relu(tmp)
                tmp = mask.repeat(N, 1) * tmp
                logits = F.linear(tmp, F2, BF2)

            _, preds = logits.max(1)

            preds = preds.cpu()
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)

            for pred, y_ in zip(preds, y):
                acc_per_class_N[y_] += 1
                if pred == y_:
                    acc_per_class_N_correct[y_] += 1

        total_acc += float(num_correct) / num_samples * 100
        total_acc_vec += acc_per_class_N_correct / acc_per_class_N

    accuracy1 = total_acc / num_run
    acc_vec_1 = total_acc_vec / num_run
        



    acc_per_class_N = np.zeros(10)
    acc_per_class_N_correct = np.zeros(10)


    mask = torch.zeros((1, 64)) + 1
    mask = mask.to(device)
    for i in range(m):
        mask[0][w[i][1]] = 0

    for X, y in loader_test:

        N, _, _, _ = X.shape


        with torch.no_grad():

            X = X.to(device)
            tmp = F.conv2d(X, C1, BC1)
            tmp = F.max_pool2d(F.relu(tmp), 2)
            tmp = F.conv2d(tmp, C2, BC2)
            tmp = F.max_pool2d(F.relu(tmp), 2)
            tmp = tmp.view(-1, 160)
            tmp = F.linear(tmp, F1, BF1)
            tmp = F.relu(tmp)
            tmp = mask.repeat(N, 1) * tmp
            logits = F.linear(tmp, F2, BF2)

        _, preds = logits.max(1)

        preds = preds.cpu()
        num_correct += (preds == y).sum()
        num_samples += preds.size(0)

        for pred, y_ in zip(preds, y):
            acc_per_class_N[y_] += 1
            if pred == y_:
                acc_per_class_N_correct[y_] += 1

    accuracy2 = float(num_correct) / num_samples * 100
    acc_vec_2 = acc_per_class_N_correct / acc_per_class_N
    # print('\nAccuracy = %.2f%%' % accuracy2)
    # print(acc_per_class_N_correct / acc_per_class_N)
    # print()

    return accuracy1, accuracy2, acc_vec_1, acc_vec_2



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

