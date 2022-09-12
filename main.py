from __future__ import print_function
from __future__ import division
from builtins import range
from builtins import int
from builtins import dict


import argparse
import copy
import pickle
import ssl
from pathlib import Path
from PIL import Image
from tqdm import tqdm


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

import torchvision
import torchvision.datasets as dset
import torchvision.models as models

from torch.utils.data import DataLoader

import numpy as np
import os

import argparse

import shutil






import warnings
warnings.filterwarnings("ignore")



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

def save_image_tensor(X, filename, norm=False):
    X_vis = X.detach().cpu().clone()
    if norm:
        X_vis[:, 0, :, :] = (X_vis[:, 0, :, :]  * 0.229) + 0.485
        X_vis[:, 1, :, :] = (X_vis[:, 1, :, :]  * 0.224 ) + 0.456
        X_vis[:, 2, :, :] = (X_vis[:, 2, :, :]  * 0.225) + 0.405
    Tr = T.ToPILImage()
    X_PIL = Tr(X_vis[0])
    # display(X_PIL)
    X_PIL.save("output_image/" + filename)

def my_vis(X, low=0, high=1, ths=None):
    
    if ths != None:
        low_th_p, high_th_p, low_th_n, high_th_n = ths
        
    X = np.mean(X, axis=0, keepdims=True)
    X_pos = np.clip(X, a_min=0, a_max=None)
    X_neg = -np.clip(X, a_min=None, a_max=0)
    
    if ths == None:
        low_th_p = np.quantile(X_pos, low)
        high_th_p = np.quantile(X_pos, high)
    X_pos = (X_pos - low_th_p) / (high_th_p - low_th_p)
    X_pos[X_pos > 1] = 1
    X_pos[X_pos < 0] = 0
    tmp = np.zeros_like(X)
    X_pos = np.concatenate([tmp, X_pos, tmp], axis=0)
    
    if ths == None:
        low_th_n = np.quantile(X_neg, low)
        high_th_n = np.quantile(X_neg, high)
    X_neg = (X_neg - low_th_n) / (high_th_n - low_th_n)
    X_neg[X_neg > 1] = 1
    X_neg[X_neg < 0] = 0
    X_neg = np.concatenate([X_neg, tmp, tmp], axis=0)
    
    return X_pos, X_neg, [low_th_p, high_th_p, low_th_n, high_th_n]



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str,
                        help="path to the image file to be computed")
    parser.add_argument("y", type=int,
                        help="true label in ImageNet dataset")
    parser.add_argument("model_first_part", type=str,
                        help="path to the first part of the model")
    parser.add_argument("model_second_part", type=str,
                        help="path to the second part of the model")
    parser.add_argument("x1", type=int,
                        help="x coordinate of the up left corner of the box")
    parser.add_argument("y1", type=int,
                        help="y coordinate of the up left corner of the box")
    parser.add_argument("x2", type=int,
                        help="x coordinate of the bottom right corner of the box")
    parser.add_argument("y2", type=int,
                        help="y coordinate of the bottom right corner of the box")

    parser.add_argument("-n", "--num_steps", type=int, default=256,
                        help="number of steps performed during the approximation")

    parser.add_argument("-p", "--percent", type=float, default=1,
                        help="# percent of neurons being pruned")

    args = parser.parse_args()
    
    
    dir = "output_image"
    if not os.path.exists(dir):
        os.makedirs(dir)

        #     shutil.rmtree(dir)


    # device = torch.device("mps")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # model = torchvision.models.resnet152(pretrained=True)
    # model_s = list(model.children())

    # model_first = nn.Sequential(*(model_s[:6]))
    # model_second = nn.Sequential(model_s[6], model_s[7], model_s[8], nn.Flatten(), model_s[9])

    model_first = torch.load(args.model_first_part)
    model_second = torch.load(args.model_second_part)

    model_first = model_first.to(device)
    model_second = model_second.to(device)

    model_first.eval()
    model_second.eval()

    for param in model_first.parameters():
        param.requires_grad = False

    for param in model_second.parameters():
        param.requires_grad = False


    dummy_input = torch.zeros((1, 3, 224, 224)).to(device)

    with torch.no_grad():
        dummy_inter = model_first(dummy_input)


    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), normalize])


    path = Path(args.image_path)

    X = torch.unsqueeze(transform(Image.open(path)), dim=0).to(device)
    
    y = torch.tensor([args.y]).to(device)

#     y = torch.tensor([920]).to(device)



    X = X.to(device)
    y = y.to(device)


    N = len(X)


    X_zero = torch.zeros(X.shape).to(device)

    X_zero[:, 0, :, :] = (X_zero[:, 0, :, :] - 0.485) / 0.229
    X_zero[:, 1, :, :] = (X_zero[:, 1, :, :] - 0.456) / 0.224
    X_zero[:, 2, :, :] = (X_zero[:, 2, :, :] - 0.405) / 0.225


    steps = args.num_steps


    delta = (X - X_zero) / steps
    kk = 0

    X_prev = X_zero

    with torch.no_grad():
        G_prev = model_first(X_prev)

    IG = torch.zeros(dummy_inter.shape, device=device)


    a, b, c, d  = args.x1, args.y1, args.x2, args.y2


    print("Computing partial IG...")
    for i in tqdm(range(steps)):

        X_p = (i + 1) * delta + X_zero

        mask = torch.zeros_like(X)

        
        mask[:, :, a: c, b: d] = 1
      

        X_p_partial = X_zero + i * delta + delta * mask


        with torch.no_grad():
            G_current = model_first(X_p)
            G_current_partial = model_first(X_p_partial)

     
        diff_G = G_current_partial - G_prev

        tmp = G_current.clone().detach()
        tmp.requires_grad = True
        logits = model_second(tmp)
        
        probs = F.softmax(logits, dim=1)
        
        # probs = probs[torch.arange(N), y]
        probs = probs.gather(1, y.unsqueeze(1))


        loss = probs.sum()

        firstterm = torch.autograd.grad(loss, tmp, only_inputs=True)[0]

        prod = firstterm * diff_G
        IG += prod

        G_prev = G_current





    pickle.dump(IG.clone().detach().cpu(), open("ig_data/tl_in_box.pkl", "wb"))



    m1 = []
    e1 = []
    m2 = []
    e2 = []

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), normalize])

    path = Path("n06874185_1258.JPEG")

    X = torch.unsqueeze(transform(Image.open(path)), dim=0).to(device)
    label = torch.tensor([920]).to(device)


    with torch.no_grad():
        logits = model_second(model_first(X))
        
    # probs = F.softmax(logits, dim=1).flatten()
    # print(probs[label])
    c2p = F.softmax(logits, dim=1).gather(1, label.unsqueeze(1)).squeeze().item()



    X_zero = torch.zeros_like(X)
    X_zero[:, 0, :, :] = (X_zero[:, 0, :, :] - 0.485) / 0.229
    X_zero[:, 1, :, :] = (X_zero[:, 1, :, :] - 0.456) / 0.224
    X_zero[:, 2, :, :] = (X_zero[:, 2, :, :] - 0.405) / 0.225



    steps = args.num_steps
    delta = (X - X_zero) / steps
    grad_acc = torch.zeros_like(X)
    print("\nComputing full IG...")
    for i in tqdm(range(steps)):

        X_p = (i + 1) * delta + X_zero
        

        X_p.requires_grad = True
        

        tmp = model_second(model_first(X_p))
        loss = F.softmax(tmp, dim=1).gather(1, label.unsqueeze(1))
        # loss = F.softmax(tmp, dim=1).flatten()[label]

        loss.backward()
        grad_acc += X_p.grad
        
        X_p.grad.zero_()
        
    IG = grad_acc * delta
    # print("IG sums up to", IG.sum().item())

    save_image_tensor(X, "raw.png", norm=True)

    # raise KeyboardInterrupt

    # X_tmp = X[:16]
    # X_tmp = X_tmp.repeat(1, 3, 1, 1)
    # print(X_tmp.shape)
    # a, b, c, d = 40, 100, 65, 130
    # a, b, c, d = 45, 100, 90, 130

    X_tmp = X.clone()


    X_tmp[:, 0, :, :] = (X_tmp[:, 0, :, :]  * 0.229) + 0.485
    X_tmp[:, 1, :, :] = (X_tmp[:, 1, :, :]  * 0.224 ) + 0.456
    X_tmp[:, 2, :, :] = (X_tmp[:, 2, :, :]  * 0.225) + 0.405
    # for i in range(16):

    X_raw = X_tmp.clone()


    for yy in range(b, d + 1):
        X_tmp[0][0][a][yy] = 1
        X_tmp[0][1][a][yy] = 1
        X_tmp[0][2][a][yy] = 0

        X_tmp[0][0][c][yy] = 1
        X_tmp[0][1][c][yy] = 1
        X_tmp[0][2][c][yy] = 0

    for xx in range(a, c + 1):
        X_tmp[0][0][xx][b] = 1
        X_tmp[0][1][xx][b] = 1
        X_tmp[0][2][xx][b] = 0

        X_tmp[0][0][xx][d] = 1
        X_tmp[0][1][xx][d] = 1
        X_tmp[0][2][xx][d] = 0
        
    X_bd = X_tmp.clone()
        
    save_image_tensor(X_tmp, "raw_box.png", norm=False)




    IG_compute = IG[0].detach().cpu().clone()
    IG_s = IG_compute.sum()
    IG_in = IG_compute[:, a: c, b: d].sum()
    IG_out = IG_s - IG_in
    print("Confidence", c2p)
    print("IG sums up to", IG_s.item())
    print("IG inside sums up to", IG_in.item())
    print("IG outside sums up to", IG_out.item())


    img = X[0].detach().cpu().clone().numpy()
    IG_tmp = IG[0].detach().cpu().clone().numpy()

    pos_IG, neg_IG, ths = my_vis(IG_tmp, low=0.8500, high=0.9999, ths=None)
    pos_IG = np.transpose(pos_IG, (1, 2, 0))
    neg_IG = np.transpose(neg_IG, (1, 2, 0))


    X_raw_ = X_raw[0].cpu()
    X_raw_ = X_raw_.mean(dim=0, keepdim=True).repeat(3, 1, 1).numpy()
    X_raw_ = np.transpose(X_raw_, (1, 2, 0))

    X_bd_ = X_bd[0].cpu()
    X_bd_ = X_bd_.mean(dim=0, keepdim=True).repeat(3, 1, 1).numpy()
    X_bd_ = np.transpose(X_bd_, (1, 2, 0))


    blend = pos_IG * 0.5 + neg_IG * 0.5 + X_raw_ * 0.5
    blend_bd = pos_IG * 0.5 + neg_IG * 0.5 + X_bd_ * 0.5


    pos_IG = Image.fromarray(np.uint8(255 * pos_IG))
    # display(pos_IG)
    pos_IG.save("output_image/IG_pos.png")


    neg_IG = Image.fromarray(np.uint8(255 * neg_IG))
    # display(neg_IG)
    neg_IG.save("output_image/IG_neg.png")


    blend = Image.fromarray(np.uint8(255 * blend))
    # display(blend)
    blend.save("output_image/IG_overlay.png")


    blend_bd = Image.fromarray(np.uint8(255 * blend_bd))
    # display(blend_bd)
    blend_bd.save("output_image/IG_overlay_box.png")

    # print("????")





    IG = []

    p = args.percent / 100


    print("\nPruning %.2f %% of the internal neurons..." % (p * 100))

    with open("ig_data/tl_in_box.pkl", "rb") as f:
        tmp = pickle.load(f)
        tmp = tmp.flatten()    
    for i, val in enumerate(tmp):
        IG.append([i, val])
    # with open("tl_out_box.pkl", "rb") as f:
    #     tmp = pickle.load(f)
    #     tmp = tmp.flatten()
    # for i, val in enumerate(tmp):
    #     IG[i][1] -= val
    IG.sort(key=lambda x: x[1], reverse=True)
    mask_ = torch.ones(dummy_inter.numel())
    for i in range(int(len(IG) * p)):
        mask_[IG[i][0]] = 0
    mask_ = mask_.view(dummy_inter.shape)
    mask_ = mask_.to(device)
        

    with torch.no_grad():
        logits = model_second(mask_ * model_first(X_p))
        
    # probs = F.softmax(logits, dim=1).flatten()
    # print(probs[label])
    c2p = F.softmax(logits, dim=1).gather(1, label.unsqueeze(1)).squeeze().item()

    print("Computing full IG after pruning...")
    steps = args.num_steps
    grad_acc = torch.zeros_like(X)
    for i in tqdm(range(steps)):

        X_p = (i + 1) * delta + X_zero
        

        X_p.requires_grad = True
        
        # X_p.grad.zero_()

        tmp = model_second(mask_ * model_first(X_p))
        loss = F.softmax(tmp, dim=1).gather(1, label.unsqueeze(1))
        # loss = F.softmax(tmp, dim=1).flatten()[label]
        loss.backward()
        grad_acc += X_p.grad
        
        X_p.grad.zero_()
        
    IG = grad_acc * delta


    IG_compute = IG[0].detach().cpu().clone()
    IG_s = IG_compute.sum()
    IG_in = IG_compute[:, a: c, b: d].sum()
    IG_out = IG_s - IG_in
    # print("IN", IG_in, "OUT", IG_out)
    print("Confidence after pruning", c2p)
    print("IG sums up to", IG_s.item())
    print("IG inside sums up to", IG_in.item())
    print("IG outside sums up to", IG_out.item())


    img = X[0].detach().cpu().clone().numpy()
    IG_tmp = IG[0].detach().cpu().clone().numpy()

    pos_IG, neg_IG, _ = my_vis(IG_tmp, ths=ths)
    pos_IG = np.transpose(pos_IG, (1, 2, 0))
    neg_IG = np.transpose(neg_IG, (1, 2, 0))


    X_raw = X_raw[0].cpu()
    X_raw = X_raw.mean(dim=0, keepdim=True).repeat(3, 1, 1).numpy()
    X_raw = np.transpose(X_raw, (1, 2, 0))

    X_bd = X_bd[0].cpu()
    X_bd = X_bd.mean(dim=0, keepdim=True).repeat(3, 1, 1).numpy()
    X_bd = np.transpose(X_bd, (1, 2, 0))


    blend = pos_IG * 0.5 + neg_IG * 0.5 + X_raw * 0.5
    blend_bd = pos_IG * 0.5 + neg_IG * 0.5 + X_bd * 0.5


    pos_IG = Image.fromarray(np.uint8(255 * pos_IG))
    # display(pos_IG)
    pos_IG.save("output_image/after_IG_pos.png")

    neg_IG = Image.fromarray(np.uint8(255 * neg_IG))
    # display(neg_IG)
    neg_IG.save("output_image/after_IG_neg.png")

    blend = Image.fromarray(np.uint8(255 * blend))
    # display(blend)
    blend.save("output_image/after_IG_overlay.png")

    blend_bd = Image.fromarray(np.uint8(255 * blend_bd))
    # display(blend_bd)
    blend_bd.save("output_image/after_IG_overlay_box.png")



    
