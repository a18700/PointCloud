#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM
"""

import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import pdb

from .attention_memchk import AttentionConv
from torch_cluster import knn_graph
import time

def memchk(i, string=None, level=0, tic=None):

    toc = time.time()
    elapsed = round((toc - tic)*1000, 3)
   
    if string is None:
        print("{} : {} GB, {} ms".format(str(i) + ' '*level, round(torch.cuda.memory_allocated(0)/1024**3, 3), elapsed))
    else:
        print("{} : {} GB, {} ms, {}".format(str(i) + ' '*level, round(torch.cuda.memory_allocated(0)/1024**3, 3), elapsed, ' (' + string + ')'))
    
    i += 1
     
    
    return i, toc


def knn(x, k):

    cnt = 0

    #MEMCHK#
    tic = time.time()
    cnt, tic = memchk(cnt, string="kNN Start", level=8, tic=tic)
    #MEMCHK#

    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]        # (batch_size, num_points, k)

    #print(x.shape)
    #print(inner.shape)
    #print(xx.shape)
    #print(pairwise_distance.shape)
    #print(idx.shape)

    #MEMCHK#
    cnt, tic = memchk(cnt, string="kNN End", level=8, tic=tic)
    #MEMCHK#

    return idx


def sparse_knn(x, k):

    cnt = 0

    #MEMCHK#
    tic = time.time()
    cnt, tic = memchk(cnt, string="kNN Start", level=8, tic=tic)
    #MEMCHK#

    # x : b, f, n
    batch, feat, npoints = x.size()
    #print("batch : {}".format(batch))
    #print("feat : {}".format(feat))
    #print("npoints : {}".format(npoints))

    x = x.permute(0, 2, 1).contiguous()  # B,N,F
    x = x.view(batch*npoints, -1) # BN, F
    #print(x.shape)12 : 4.827 GB (conv4)


    knn_batch = torch.tensor([[i]*npoints for i in range(batch)], device='cuda').flatten() # BN
    #print(knn_batch.shape)

    edge_index = knn_graph(x, k=k, batch=knn_batch, loop=False) # BNK
    #print(edge_index[0].shape)
    #print(edge_index[1].shape)
    idx = edge_index[0].view(batch, npoints, k)

    idx_base = torch.arange(0, batch, device=x.device).view(-1, 1, 1)*npoints # 1, 1, 1
    idx = idx - idx_base

    #print(knn_batch[1000:1100])
    #print(edge_index[0][1024*20-100:1024*20+100])
    #print(edge_index[1][1024*20-100:1024*20+100])

    #assert 1==2, "stop"
    #MEMCHK#
    cnt, tic = memchk(cnt, string="kNN End", level=8, tic=tic)
    #MEMCHK#


    return idx


def get_neighbors(x, k=20, dim9=False):

    cnt = 0
    
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    abs_x = x.unsqueeze(3)

    #MEMCHK#
    tic = time.time()
    cnt, tic = memchk(cnt, level=4, tic=tic)
    #MEMCHK#


    if dim9 == False:
        #idx = knn(x, k=k)   # (batch_size, num_points, k)
        idx = sparse_knn(x, k=k)   # (batch_size, num_points, k)
        idx_return = idx.unsqueeze(1)   # (batch, 1, num_points, k)
    else:
        #idx = knn(x[:, 6:], k=k)
        idx = sparse_knn(x[:, 6:], k=k)
        idx_return = idx.unsqueeze(1)   # (batch, 1, num_points, k)

    #MEMCHK#
    cnt, tic = memchk(cnt, level=4, tic=tic)
    #MEMCHK#

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1)*num_points # 1, 1, 1
    idx = idx + idx_base        # B, N, K
    idx = idx.view(-1)

    _, num_dims, _ = x.size()   # 3
    x = x.transpose(2, 1).contiguous()      # B, N, C

    feature = x.view(batch_size*num_points, -1)[idx, :] # (points*k*batch_size, 3)
    feature = feature.view(batch_size, num_points, k, -1) # batch, N, K, C
    x = x.view(batch_size, num_points, 1, -1).repeat(1, 1, k, 1) # batch, N, K, C

    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2)  # batch, C, N , K
    return feature, abs_x, idx_return


class DGCNN_Transformer(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNN_Transformer, self).__init__()

        self.args = args
        self.k = args.k
        self.ape = args.ape
        self.scale = args.scale
        print("self.ape : {}".format(self.ape))
        print("self.scale : {}".format(self.scale))

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        
        self.conv1 = AttentionConv(3*2, 64, kernel_size=self.k, groups=8, ape=self.ape, scale=self.scale)
        self.act1 = nn.LeakyReLU(negative_slope=0.2)
        self.conv2 = AttentionConv(64*2, 64, kernel_size=self.k, groups=8, ape=self.ape, scale=self.scale)
        self.act2 = nn.LeakyReLU(negative_slope=0.2)
        self.conv3 = AttentionConv(64*2, 128, kernel_size=self.k, groups=8, ape=self.ape, scale=self.scale)
        self.act3 = nn.LeakyReLU(negative_slope=0.2)
        self.conv4 = AttentionConv(128*2, 256, kernel_size=self.k, groups=8, ape=self.ape, scale=self.scale)
        self.act4= nn.LeakyReLU(negative_slope=0.2)

        self.conv5 = nn.Sequential(nn.Conv1d(256, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2)) # 64 + 64 + 128 + 256 = 512

        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

        for key in self.state_dict():
            if key.split('.')[-1]=="weight":
                try:
                    if "conv" in key:
                        init.kaiming_normal(self.state_dict()[key])
                except:
                    init.normal(self.state_dict()[key])
                if "bn" in key:
                    self.state_dict()[key][...] = 1
            elif key.split(".")[-1]=='bias':
                self.state_dict()[key][...] = 0


    def forward(self, x):

        batch_size = x.size(0)

        x = x.repeat(1, 1, 8)

        #print(x.shape)

        # shape of x : batch, feature, npoints, neighbors
        # =>
        # transformer(shared wq, wk, wv to xi)

        i = 1
        points = x
   
        tic = time.time()
        tic_init = tic
        i, tic = memchk(i, string="points", tic=tic)
        x, abs_x, idx1 = get_neighbors(x, k=self.k) # b, 64, 1024, 20
        i, tic = memchk(i, string="kNN1", tic=tic)
        x1 = self.conv1(x, abs_x, idx1, points) # b, 64, 1024
        i, tic = memchk(i, string="conv1", tic=tic)
        x1 = self.act1(self.bn1(x1)).squeeze(3)
        i, tic = memchk(i, string="bnrelu1", tic=tic)

        x, abs_x, idx2 = get_neighbors(x1, k=self.k) # b, 64, 1024, 20
        i, tic = memchk(i, string="kNN2", tic=tic)
        x2 = self.conv2(x, abs_x, idx2, points) # b, 64, 1024
        i, tic = memchk(i, string="conv2", tic=tic)
        x2 = self.act2(self.bn2(x2)).squeeze(3)
        i, tic = memchk(i, string="bnrelu2", tic=tic)

        x, abs_x, idx3 = get_neighbors(x2, k=self.k) # b, 64, 1024, 20
        i, tic = memchk(i, string="kNN3", tic=tic)
        x3 = self.conv3(x, abs_x, idx3, points) # b, 128, 1024
        i, tic = memchk(i, string="conv3", tic=tic)
        x3 = self.act3(self.bn3(x3)).squeeze(3)
        i, tic = memchk(i, string="bnrelu3", tic=tic)

        x, abs_x, idx4 = get_neighbors(x3, k=self.k) # b, 64, 1024, 20
        i, tic = memchk(i, string="kNN4", tic=tic)
        x4 = self.conv4(x, abs_x, idx4, points) # b, 256, 1024, 20
        i, tic = memchk(i, string="conv4", tic=tic)
        x4 = self.act4(self.bn4(x4)).squeeze(3)
        i, tic = memchk(i, string="bnrelu4", tic=tic)

        x = self.conv5(x4)
        i, tic = memchk(i, string="conv5", tic=tic) # 8
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        i, tic = memchk(i, string="dp1", tic=tic) # 8
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        i, tic = memchk(i, string="dp2", tic=tic) # 8
        x = self.linear3(x)
        i, tic_final = memchk(i, string="fc", tic=tic) # 8

        print("Total elapsed time : {} ms".format(round((tic_final - tic_init)*1000,3)))

        return x



