from cmath import log
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim
import os
from numpy import argmax, array, zeros, full, argmin, inf, ndim
from math import isinf
from scipy.spatial.distance import cdist
from numba import jit
import sys
sys.path.append(r"/home/denglong/code/FSL-Video")
from utils import parse_args, model_dict
# from dataset_23 import SetDataManager_RGB_FLOW
from dataset_8 import SetDataManager_RGB_FLOW  #随机
from meta_template import BaseNet
from Kuhn_Munkres_Algorithm.km import run_kuhn_munkres
from sklearn.cluster import KMeans
from pykeops.torch import LazyTensor
from cmath import log

def KMeans_cosine(x, K=10, Niter=10, verbose=True):
    N, D = x.shape 
    c = x[:K, :].clone()  
    c = torch.nn.functional.normalize(c, dim=1, p=2)
    x_i = LazyTensor(x.view(N, 1, D))  
    c_j = LazyTensor(c.view(1, K, D))  
    for i in range(Niter):
        S_ij = x_i | c_j  
        cl = S_ij.argmax(dim=1).long().view(-1) 
        c.zero_()
        c.scatter_add_(0, cl[:, None].repeat(1, D), x)
        c[:] = torch.nn.functional.normalize(c, dim=1, p=2)
    return cl, c

def train_epoch(logits):
    nway, sq, t, out = logits.shape
    logits = logits.reshape(nway, sq, t, logits.shape[-1]) 
    if isinstance(model, nn.DataParallel):
        n_support = model.module.n_support
    else:
        n_support = model.n_support

    n_query = sq - n_support
    z_support = logits[:, :n_support].reshape(nway * n_support, t, -1)
    z_query = logits[:, n_support:].reshape(nway * n_query, t, -1)
    scores = torch.zeros((nway * n_query, nway * n_support), requires_grad=True).cuda()
    for qi in range(nway * n_query):  
        for si in range(nway * n_support):  
            x = z_support[si].cpu().detach().numpy()
            y = z_query[qi].cpu().detach().numpy()
            _, path = dtw(x, y)

            start = np.where(path[0] == 1)[0][0]
            x_index = path[0][start:start + t] - 1  
            y_index = path[1][start:start + t]
            scores[qi][si] = F.cosine_similarity(z_support[si][x_index], z_query[qi][y_index]).sum()

            _, path = dtw(y, x)
            start = np.where(path[0] == 1)[0][0]
            x_index = path[0][start:start + t] - 1  
            y_index = path[1][start:start + t]
            scores[qi][si] += F.cosine_similarity(z_support[si][y_index], z_query[qi][x_index]).sum()
    scores = scores.reshape(nway * n_query, nway, n_support).mean(2)
    y_query = torch.from_numpy(np.repeat(range(nway), n_query))
    y_query = Variable(y_query.cuda())
    return scores, y_query

# def kmeans_cluster(features):
#     num_clusters = 3  

#     features_np = features.cpu().numpy()

#     kmeans = KMeans(n_clusters=num_clusters, random_state=0)
#     cluster_centers = kmeans.fit_predict(features_np)

#     new_features = torch.tensor(kmeans.cluster_centers_[cluster_centers])

#     new_features = new_features.view(8, 2048)
#     return new_features

def train_epoch_km(logits):
    nway, sq, t, out = logits.shape
    num_clusters = 3
    Niter = 10

    logits = logits.reshape(nway, sq, t, logits.shape[-1]) 
    if isinstance(model, nn.DataParallel):
        n_support = model.module.n_support
    else:
        n_support = model.n_support

    n_query = sq - n_support
    z_support = logits[:, :n_support].reshape(nway * n_support, t, -1)
    z_query = logits[:, n_support:].reshape(nway * n_query, t, -1)
    scores_km = torch.zeros((nway * n_query, nway * n_support), requires_grad=True).cuda()
    for qi in range(nway * n_query): 
        for si in range(nway * n_support):  
            x = z_support[si].cuda().detach()
            y = z_query[qi].cuda().detach()

            cl, c = KMeans_cosine(x, num_clusters, Niter)
            x_new = c[cl]
            cl, c = KMeans_cosine(y, num_clusters, Niter)
            y_new = c[cl]
            x_norm = torch.norm(x_new, p=2, dim=1).reshape(-1, 1)
            y_norm = torch.norm(y_new, p=2, dim=1).reshape(1, -1)
            sim_km = torch.matmul(x_new, y_new.t())/(torch.matmul(x_norm, y_norm))
            temp1 = torch.arange(1, t+1, 1)
            temp2 = temp1.repeat((1, t)).reshape(t*t,1).cuda()
            temp3 = torch.arange(1, t+1, 1).reshape(1, t)
            temp4 = temp3.repeat((t,1)).t().reshape(-1,1).cuda()
            sim_km1 = sim_km.reshape(t*t,1)
            value_list = torch.cat([temp4,temp2,sim_km1],dim=1).tolist()
            values = [tuple(sublist) for sublist in value_list]
            match_result = run_kuhn_munkres(values)
            for match in match_result:
                similarity = match[2]
                scores_km[qi][si] += similarity
    scores_km = scores_km.reshape(nway * n_query, nway, n_support).mean(2)
    y_query = torch.from_numpy(np.repeat(range(nway), n_query))
    y_query = Variable(y_query.cuda())
    return scores_km, y_query

def get_mask(n_shot):
    if n_shot==5:
        mask = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

        # mask = [[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #         [1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #         [1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
        #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #         [1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #         [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
        #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
        #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #          1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #          0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #          0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #          0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #          0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #          0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #          0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #          0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #          0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #          0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        #         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0,
        #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0,
        #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0,
        #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0,
        #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0,
        #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
        #          1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1,
        #          1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1,
        #          1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0,
        #          1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
        #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #          0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #          0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #          0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #          0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #          0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #          0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #          0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #          0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #          0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #          0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1],
        #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1],
        #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1],
        #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]]


    else:
        mask = [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]

    return mask


def train(data_loader_list, model, flow_model, optimization, start_epoch, stop_epoch, params):
    [base_loader,  val_loader, test_loader] = data_loader_list

    trainfile_path = ""
    if not os.path.isdir(trainfile_path):
        os.makedirs(trainfile_path)
    log_file = open(trainfile_path+"/train_log.txt","a+")
    print(trainfile_path)

    if optimization == 'Adam':
        lr = params.lr
        optimizer_rgb = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError('Unknown optimization, please define by yourself')
    if optimization == 'Adam':
        lr = params.lr
        optimizer_flow = torch.optim.Adam(flow_model.parameters(), lr=lr)
    else:
        raise ValueError('Unknown optimization, please define by yourself')

    def post_process(scores, y_query, acc_all, isSoftmax = False):
        if isSoftmax:
            scores = F.softmax(scores, dim=0)
        topk_scores_rgb, topk_labels = scores.data.topk(1, 1, True, True)

        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:, 0] == y_query.cpu().numpy())
        correct_this, count_this = float(top1_correct), len(y_query)
        acc_all.append(correct_this / count_this * 100)

        return scores

    max_acc = 0
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(start_epoch, stop_epoch):
        model.train()
        flow_model.train()
        avg_loss_rgb = 0
        avg_loss_flow = 0
        avg_loss_rgb_km = 0
        avg_loss_flow_km = 0

        for i, (x, _) in enumerate(base_loader):
            x = x.cuda()
            nway, sq, t, c, h, w = x.shape

            x_gray = x[:, :, :, 3:6, :, :]
            x_gray = x_gray.reshape(nway * sq * t, 3, h, w)
            gray_logits_all = model(x_gray)
            gray_logits_all = gray_logits_all.reshape(nway, sq, t, gray_logits_all.shape[-1])

            x1_img = x[:, :, :, 6, :, :]
            y1_img = x[:, :, :, 7, :, :]
            x2_img = x[:, :, :, 8, :, :]
            y2_img = x[:, :, :, 9, :, :]
            x_flow = torch.cat([x1_img, y1_img, x2_img, y2_img], dim=3)
            x_flow = x_flow.reshape(nway, sq, t, 4, h, w)
            x_flow = x_flow.reshape(nway * sq * t, 4, h, w)
            flow_logits_all = flow_model(x_flow)
            flow_logits_all = flow_logits_all.reshape(nway, sq, t, flow_logits_all.shape[-1])

            rgb_logits_all = gray_logits_all.reshape(nway, sq, t, 2048)
            flow_logits_all = flow_logits_all.reshape(nway, sq, t, 2048)
            scores_rgb_km, y_query_rgb_km = train_epoch_km(rgb_logits_all)
            scores_flow_km, y_query_flow_km = train_epoch_km(flow_logits_all)
            scores_rgb, y_query_rgb = train_epoch(rgb_logits_all)
            scores_flow, y_query_flow = train_epoch(flow_logits_all)

            loss_rgb_km = loss_fn(scores_rgb_km, y_query_rgb)
            loss_flow_km = loss_fn(scores_flow_km, y_query_flow)
            loss_rgb = loss_fn(scores_rgb, y_query_rgb)
            loss_flow = loss_fn(scores_flow, y_query_flow)
            loss = loss_rgb + loss_flow + loss_rgb_km + loss_flow_km
            optimizer_rgb.zero_grad()
            optimizer_flow.zero_grad()
            loss.backward()
            optimizer_rgb.step()
            optimizer_flow.step()

            avg_loss_rgb = avg_loss_rgb + loss_rgb.item()
            avg_loss_flow = avg_loss_flow + loss_flow.item()
            avg_loss_rgb_km = avg_loss_rgb_km + loss_rgb_km.item()
            avg_loss_flow_km = avg_loss_flow_km + loss_flow_km.item()

        print('Epoch {:d} |loss_rgb {:f} | loss_flow {:f}| loss_rgb_km {:f} | loss_flow_km {:f}\n'.format(epoch,avg_loss_rgb / float(i + 1), avg_loss_flow / float(i + 1),avg_loss_rgb_km / float(i + 1), avg_loss_flow_km / float(i + 1)),end="",file = log_file,flush=True)

        if epoch > 100:
            model.eval()
            flow_model.eval()
            if not os.path.isdir(params.checkpoint_dir):
                os.makedirs(params.checkpoint_dir)

            acc_all = []
            acc_all_rgb = []
            acc_all_flow = []
            acc_all_rgb_km = []
            acc_all_flow_km = []
            iter_num = len(val_loader)
            with torch.no_grad():
                for i, (x, _) in enumerate(val_loader):
                    x = x.cuda()
                    nway, sq, t, c, h, w = x.shape

                    x_gray = x[:, :, :, 3:6, :, :]
                    x_gray = x_gray.reshape(nway * sq * t, 3, h, w)
                    gray_logits_all = model(x_gray)

                    x1_img = x[:, :, :, 6, :, :]
                    y1_img = x[:, :, :, 7, :, :]
                    x2_img = x[:, :, :, 8, :, :]
                    y2_img = x[:, :, :, 9, :, :]
                    x_flow = torch.cat([x1_img, y1_img, x2_img, y2_img], dim=3)
                    x_flow = x_flow.reshape(nway, sq, t, 4, h, w)
                    x_flow = x_flow.reshape(nway * sq * t, 4, h, w)
                    flow_logits = flow_model(x_flow)
                    rgb_logits = gray_logits_all.reshape(nway, sq, t, 2048)
                    flow_logits = flow_logits.reshape(nway, sq, t, 2048)
                    scores_rgb_km, y_query_rgb_km = train_epoch_km(rgb_logits)
                    scores_flow_km, y_query_flow_km = train_epoch_km(flow_logits)
                    scores_rgb, y_query_rgb = train_epoch(rgb_logits)
                    scores_flow, y_query_flow = train_epoch(flow_logits)
                    scores_rgb = F.softmax(scores_rgb, dim=0)
                    scores_flow = F.softmax(scores_flow, dim=0)
                    scores_rgb_km = F.softmax(scores_rgb_km, dim=0)
                    scores_flow_km = F.softmax(scores_flow_km, dim=0)

                    scores_rgb_km = post_process(scores_rgb_km, y_query_rgb, acc_all_rgb_km)
                    scores_flow_km = post_process(scores_flow_km, y_query_flow, acc_all_flow_km)
                    scores_rgb = post_process(scores_rgb, y_query_rgb, acc_all_rgb)
                    scores_flow = post_process(scores_flow, y_query_flow, acc_all_flow)
                    scores = 0.3*(0.5*scores_rgb+0.5*scores_rgb_km) + 0.7*(0.5*scores_flow+0.5*scores_flow_km)
                    _ = post_process(scores, y_query_rgb, acc_all, False)

            acc_all_rgb_km = np.asarray(acc_all_rgb_km)
            acc_mean = np.mean(acc_all_rgb_km)
            acc_std = np.std(acc_all_rgb_km)
            print('RGB_km %d Val Acc = %4.2f%% +- %4.2f%% \n' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)), end="",file = log_file,flush=True)

            acc_all_flow_km = np.asarray(acc_all_flow_km)
            acc_mean = np.mean(acc_all_flow_km)
            acc_std = np.std(acc_all_flow_km)
            print('Flow_km %d Val Acc = %4.2f%% +- %4.2f%% \n' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)), end="",file = log_file,flush=True)


            acc_all_rgb = np.asarray(acc_all_rgb)
            acc_mean = np.mean(acc_all_rgb)
            acc_std = np.std(acc_all_rgb)
            print('RGB %d Val Acc = %4.2f%% +- %4.2f%% \n' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)), end="",file = log_file,flush=True)

            acc_all_flow = np.asarray(acc_all_flow)
            acc_mean = np.mean(acc_all_flow)
            acc_std = np.std(acc_all_flow)
            print('Flow %d Val Acc = %4.2f%% +- %4.2f%% \n' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)), end="",file = log_file,flush=True)

            acc_all2 = np.asarray(acc_all)
            acc_mean2 = np.mean(acc_all2)
            acc_std2 = np.std(acc_all2)
            print('Both %d Val Acc = %4.2f%% +- %4.2f%% \n' % (iter_num, acc_mean2, 1.96 * acc_std2 / np.sqrt(iter_num)), end="",file = log_file,flush=True)

            if acc_mean > max_acc:
                print("best model! save...", file=log_file, flush=True)
                max_acc = acc_mean
                outfile = os.path.join(trainfile_path, 'best_rgbmodel.tar')
                torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)
                outfile = os.path.join(trainfile_path, 'best_flowmodel.tar')
                torch.save({'epoch': epoch, 'state': flow_model.state_dict()}, outfile)

            print('best Acc = %4.2f%% +- %4.2f%% \n' % (max_acc, 1.96 * acc_std / np.sqrt(iter_num)), file=log_file,
                    flush=True)
    return model


def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def dtw(x, y):
    r, c = len(x), len(y)
    assert r == c
    r = r + 2
    D0 = np.zeros((r + 1, c + 1))
    D0[0, 1:] = np.inf
    D0[1:, 0] = np.inf
    D1 = D0[1:, 1:]  # view

    # for i in range(1,r-1): # non zero
    # for j in range(c):
    # D1[i, j] = dist(x[i-1], y[j])

    n = x.shape[0]
    m = y.shape[0]
    d = x.shape[1]
    assert n == m
    assert d == y.shape[1]

    if False:  # Euclidean
        x = np.broadcast_to(np.expand_dims(x, 1), (n, m, d))
        y = np.broadcast_to(np.expand_dims(y, 0), (n, m, d))

        D1[1:r - 1, 0:c] = np.power(x - y, 2).sum(2)

    else:  # cosine
        D1[1:r - 1, 0:c] = cdist(x, y, 'cosine')
    # print(D1)

    D0, D1 = dtw_loop(r, c, D0, D1)

    if len(x) == 1:
        path = np.zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), np.zeros(len(x))
    else:
        path = _traceback(D0)
    # return D1[-1, -1], C, D1, path
    return D1[-1, -1], path


@jit(nopython=True)
def dtw_loop(r, c, D0, D1):
    for i in range(r):  # [0,T+1]
        for j in range(c):  # [0,T-1]
            # min_list = [D0[i, j]]
            i_k = min(i + 1, r)
            j_k = min(j + 1, c)
            if i == 0 or i == r - 1:  # first and last
                # min_list += [D0[i_k, j] * s, D0[i, j_k] * s]
                min_list = [D0[i, j], D0[i_k, j], D0[i, j_k]]
            else:
                # min_list += [D0[i, j_k] * s]
                min_list = [D0[i, j], D0[i, j_k]]
            # D1[i, j] += min(min_list) # smooth version
            # print(min_list)

            D1[i, j] += min(min_list)
    return D0, D1


def _traceback(D):
    i, j = np.array(D.shape) - 2
    p, q = [i], [j]
    while (i > 0) or (j > 0):
        tb = np.argmin((D[i, j], D[i, j + 1], D[i + 1, j]))
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        else:  # (tb == 2):
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return np.array(p), np.array(q)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    np.random.seed(10)
    params = parse_args('train')
    if params.dataset == 'kinetics':
        base_file = ''
        val_file = ''
        test_file = ''
        base_flow = ''
        val_flow = ''
        test_flow = ''
    elif params.dataset == 'somethingotam':
        base_file = ''
        val_file = ''
        test_file = ''
        base_flow = ''
        val_flow = ''
        test_flow = ''
    else:
        raise ValueError('Unknown dataset')

    image_size = 224
    optimization = 'Adam'

    if params.stop_epoch == -1:
        params.stop_epoch = 800

    params.method = 'otam'
    params.with_flow = True
    if params.method in ['otam']:
        n_query = max(1,int(params.n_query * params.test_n_way / params.train_n_way))

        #----------------------------train--------------------------------
        train_few_shot_params = dict(n_way=params.train_n_way, n_support=params.n_shot)
        base_datamgr = SetDataManager_RGB_FLOW(image_size, n_query=n_query, num_segments=params.num_segments,
                                      **train_few_shot_params)
        base_loader = base_datamgr.get_data_loader(base_file, base_flow, aug=params.train_aug)

        #----------------------------val--------------------------------
        test_few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot)
        val_datamgr = SetDataManager_RGB_FLOW(image_size, n_query=n_query, num_segments=params.num_segments,
                                     n_eposide=params.eval_episode, **test_few_shot_params)
        val_loader = val_datamgr.get_data_loader(val_file, val_flow, aug=False)

        # ----------------------------test--------------------------------
        test_datamgr = SetDataManager_RGB_FLOW(image_size, n_query=n_query, num_segments=params.num_segments,
                                      n_eposide=params.test_episode, **test_few_shot_params)
        test_loader = test_datamgr.get_data_loader(test_file, test_flow, aug=False)

        model = BaseNet(model_dict[params.model], **train_few_shot_params)
        flow_model = BaseNet(model_dict[params.flow_model], **train_few_shot_params)

    else:
        raise ValueError('Unknown method')

    model = model.cuda()
    flow_model = flow_model.cuda()

    if params.test_model:
        checkpoint = torch.load(params.checkpoint, map_location=lambda storage, loc: storage.cuda(0))
        checkpoint = checkpoint['state']
        base_dict = {}
        for k, v in list(checkpoint.items()):
            if k.startswith('module'):
                base_dict['.'.join(k.split('.')[1:])] = v
            else:
                base_dict[k] = v
        model.load_state_dict(base_dict)
        checkpoint_flow = torch.load(params.checkpoint_flow, map_location=lambda storage, loc: storage.cuda(0))
        checkpoint_flow = checkpoint_flow['state']
        base_dict = {}
        for k, v in list(checkpoint_flow.items()):
            if k.startswith('module'):
                base_dict['.'.join(k.split('.')[1:])] = v
            else:
                base_dict[k] = v
        flow_model.load_state_dict(base_dict)
        test_loader_list = [test_loader]
    else:
        model = nn.DataParallel(model)
        flow_model = nn.DataParallel(flow_model)

        params.checkpoint_dir = '%s/checkpoints/%s/%s/%s' % (
        params.work_dir, params.dataset, params.model, params.method)

        if params.train_aug:
            params.checkpoint_dir += '_aug'
        params.checkpoint_dir += '_%dway_%dshot' % (params.train_n_way, params.n_shot)

        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)

        start_epoch = params.start_epoch
        stop_epoch = params.stop_epoch
        data_loader_list = [base_loader,  val_loader, test_loader]
        model = train(data_loader_list, model, flow_model, optimization, start_epoch, stop_epoch, params)
