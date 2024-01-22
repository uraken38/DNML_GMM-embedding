import torch
import random
import numpy as np
from torch import nn
from torch import optim
from tqdm import trange, tqdm

from torch.utils.data import Dataset, DataLoader

import argparse

from copy import deepcopy
import torch.multiprocessing as multi
from functools import partial
import pandas as pd
import gc
import os
import time
from torch import Tensor
from scipy import integrate
from sklearn import metrics
import math
from scipy import stats, special

import sklearn.mixture as mixture

from numpy.linalg import det, inv
from scipy.special import logsumexp, loggamma
from scipy.stats import multivariate_normal
from scipy.spatial.distance import pdist, squareform

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


def connection_prob(d, R, beta):
    """
    接続確率
    """
    return 1 / (1 + np.exp(beta * (d - R)))

# lorentz scalar product
def distance_mat(X, Y):
    X = X[:, np.newaxis, :]
    Y = Y[np.newaxis, :, :]
    Z = np.sqrt(np.sum((X - Y) ** 2, axis=2))
    return Z

def create_dataset_for_basescore(
    adj_mat,
    n_max_samples,
):
    # データセットを作成し、trainとvalidationに分ける
    _adj_mat = deepcopy(adj_mat)
    n_nodes = _adj_mat.shape[0]
    for i in range(n_nodes):
        _adj_mat[i, i] = -1
    # -1はサンプリング済みの箇所か対角要素

    data = []

    for i in range(n_nodes):
        idx_samples = np.where(_adj_mat[i, :] != -1)[0]
        idx_samples = np.random.permutation(idx_samples)
        n_samples = min(len(idx_samples), n_max_samples)

        # node iを固定した上で、positiveなnode jを対象とする。それに対し、
        for j in idx_samples[0:n_samples]:
            data.append((i, j, _adj_mat[i, j]))


    return data

def create_test_for_link_prediction(
    adj_mat,
    params_dataset
):
    # testデータとtrain_graphを作成する
    n_total_positives = np.sum(adj_mat) / 2
    n_samples_test = int(n_total_positives * 0.1)
    n_neg_samples_per_positive = 1  # positive1つに対してnegativeをいくつサンプリングするか

    # positive sampleのサンプリング
    train_graph = np.copy(adj_mat)
    # 対角要素からはサンプリングしない
    for i in range(params_dataset["n_nodes"]):
        train_graph[i, i] = -1

    positive_samples = np.array(np.where(train_graph == 1)).T
    # 実質的に重複している要素を削除
    positive_samples_ = []
    for p in positive_samples:
        if p[0] > p[1]:
            positive_samples_.append([p[0], p[1]])
    positive_samples = np.array(positive_samples_)

    positive_samples = np.random.permutation(positive_samples)[:n_samples_test]

    # サンプリングしたデータをtrain_graphから削除
    for t in positive_samples:
        train_graph[t[0], t[1]] = -1
        train_graph[t[1], t[0]] = -1

    # negative sampleのサンプリング
    # permutationが遅くなるので直接サンプリングする
    negative_samples = []
    while len(negative_samples) < n_samples_test * n_neg_samples_per_positive:
        u = np.random.randint(0, params_dataset["n_nodes"])
        v = np.random.randint(0, params_dataset["n_nodes"])
        if train_graph[u, v] != 0:
            continue
        else:
            negative_samples.append([u, v])
            train_graph[u, v] = -1
            train_graph[v, u] = -1

    negative_samples = np.array(negative_samples)

    # これは重複を許す
    lik_data = create_dataset_for_basescore(
        adj_mat=train_graph,
        n_max_samples=int((params_dataset["n_nodes"] - 1) * 0.1)
    )

    return positive_samples, negative_samples, train_graph, lik_data

def e_dist(
    u_e,
    v_e,
    use_torch=True
):
    if use_torch:
        return torch.sqrt(torch.sum((u_e - v_e)**2, dim=1))
    else:
        return np.sqrt(np.sum((u_e - v_e)**2, axis=1))

def multigamma_ln(a, d):
    return special.multigammaln(a, d)

def _pc_multinomial(N, K):
    """parametric complexity for multinomial distributions.
    Args:
        N (int): number of data.
        K (int): number of clusters.
    Returns:
        float: parametric complexity for multinomial distributions.
    """
    PC_list = [0]

    # K = 1
    if K >= 1:
        PC_list.append(1)

    # K = 2
    if K >= 2:
        r1 = np.arange(N + 1)
        r2 = N - r1
        logpc_2 = logsumexp(sum([
            loggamma(N + 1),
            (-1) * loggamma(r1 + 1),
            (-1) * loggamma(r2 + 1),
            r1 * np.log(r1 / N + 1e-50),
            r2 * np.log(r2 / N + 1e-50)
        ]))
        PC_list.append(np.exp(logpc_2))

    # K >= 3
    for k in range(3, K + 1):
        PC_list.append(PC_list[k - 1] + N * PC_list[k - 2] / (k - 2))

    return PC_list[-1]

def _log_pc_gaussian(N_list, D, R, lmd_min):
    """log parametric complexity for Gaussian distributions.
    Args:
        N_list (np.ndarray): list of the number of data.
        D (int): dimension of data.
        R (float): upper bound of ||mean||^2.
        lmd_min (float): lower bound of the eigenvalues of the covariance matrix.
    Returns:
        np.ndarray: list of the parametric complexity.
    """
    N_list = np.array(N_list)

    log_PC_list = sum([
        D * N_list * np.log(N_list / 2 / math.e) / 2,
        (-1) * D * (D - 1) * np.log(math.pi) / 4,
        (-1) * np.sum(loggamma((N_list.reshape(-1, 1) - np.arange(1, D + 1)) / 2), axis=1),
        (D + 1) * np.log(2 / D),
        (-1) * loggamma(D / 2),
        D * np.log(R) / 2,
        (-1) * D**2 * np.log(lmd_min) / 2
    ])

    return log_PC_list


def log_pc_gmm(K_max, N_max, D, *, R=1e+3, lmd_min=1e-3):
    """log PC of GMM.
    Calculate (log) parametric complexity of Gaussian mixture model.
    Args:
        K_max (int): max number of clusters.
        N_max (int): max number of data.
        D (int): dimension of data.
        R (float): upper bound of ||mean||^2.
        lmd_min (float): lower bound of the eigenvalues of the covariance matrix.
    Returns:
        np.ndarray: array of (log) parametric complexity.
            returns[K, N] = log C(K, N)
    """
    log_PC_array = np.zeros([K_max + 1, N_max + 1])
    r1_min = D + 1

    # N = 0
    log_PC_array[:, 0] = -np.inf

    # K = 0
    log_PC_array[0, :] = -np.inf

    # K = 1
    # N <= r1_min
    log_PC_array[1, :r1_min] = -np.inf
    # N > r1_min
    N_list = np.arange(r1_min, N_max + 1)
    log_PC_array[1, r1_min:] = _log_pc_gaussian(N_list, D=D, R=R, lmd_min=lmd_min)

    # K > 1
    for k in range(2, K_max + 1):
        for n in range(1, N_max + 1):
            r1 = np.arange(n + 1)
            r2 = n - r1
            log_PC_array[k, n] = logsumexp(sum([
                loggamma(n + 1),
                (-1) * loggamma(r1 + 1),
                (-1) * loggamma(r2 + 1),
                r1 * np.log(r1 / n + 1e-100),
                r2 * np.log(r2 / n + 1e-100),
                log_PC_array[1, r1],
                log_PC_array[k - 1, r2]
            ]))

    return log_PC_array

# ロジスティック回帰モデルの対数尤度を計算する関数
def calculate_negative_log_likelihood(coefficients, intercept, X, Y):
    probabilities = 1 / (1 + np.exp(-(X * coefficients + intercept)))
    log_likelihood = np.sum(Y * np.log(probabilities) + (1 - Y) * np.log(1 - probabilities))
    return -log_likelihood



class Graph(Dataset):

    def __init__(
        self,
        data
    ):
        self.data = torch.Tensor(data).long()
        self.n_items = len(data)

    def __len__(self):
        # データの長さを返す関数
        return self.n_items

    def __getitem__(self, i):
        # ノードとラベルを返す。
        return self.data[i, 0:2], self.data[i, 2]


class NegGraph(Dataset):

    def __init__(
        self,
        adj_mat,
        n_max_positives=5,
        n_max_negatives=50,
    ):
        # データセットを作成し、trainとvalidationに分ける
        self.n_max_positives = n_max_positives
        self.n_max_negatives = n_max_negatives
        self._adj_mat = deepcopy(adj_mat)
        self.n_nodes = self._adj_mat.shape[0]
        for i in range(self.n_nodes):
            self._adj_mat[i, i] = -1

    def __len__(self):
        # データの長さを返す関数
        return self.n_nodes

    def __getitem__(self, i):

        data = []

        # positiveをサンプリング
        idx_positives = np.where(self._adj_mat[i, :] == 1)[0]
        idx_negatives = np.where(self._adj_mat[i, :] == 0)[0]
        idx_positives = np.random.permutation(idx_positives)
        idx_negatives = np.random.permutation(idx_negatives)
        n_positives = min(len(idx_positives), self.n_max_positives)
        n_negatives = min(len(idx_negatives), self.n_max_negatives)

        # node iを固定した上で、positiveなnode jを対象とする。それに対し、
        for j in idx_positives[0:n_positives]:
            data.append((i, j, 1))  # positive sample

        for j in idx_negatives[0:n_negatives]:
            data.append((i, j, 0))  # negative sample

        if n_positives + n_negatives < self.n_max_positives + self.n_max_negatives:
            rest = self.n_max_positives + self.n_max_negatives - \
                (n_positives + n_negatives)
            rest_idx = np.append(
                idx_positives[n_positives:], idx_negatives[n_negatives:])
            rest_label = np.append(np.ones(len(idx_positives) - n_positives), np.zeros(
                len(idx_negatives) - n_negatives))

            rest_data = np.append(rest_idx.reshape(
                (-1, 1)), rest_label.reshape((-1, 1)), axis=1).astype(np.int)

            rest_data = np.random.permutation(rest_data)

            for datum in rest_data[:rest]:
                data.append((i, datum[0], datum[1]))

        data = np.random.permutation(data)

        torch.Tensor(data).long()

        # ノードとラベルを返す。
        return data[:, 0:2], data[:, 2]



class SGD_Gaussian(optim.Optimizer):
    """
    Stochastic Gradient Descentを行う関数。
    """

    def __init__(
        self,
        params,
        lr_embeddings,
        lr_beta,
        lr_gamma,
        R,
        beta_max,
        beta_min,
        gamma_max,
        gamma_min,
        device
    ):
        defaults = {
            "lr_embeddings": lr_embeddings,
            "lr_beta": lr_beta,
            "lr_gamma": lr_gamma,
            'R': R,
            "beta_max": beta_max,
            "beta_min": beta_min,
            "gamma_max": gamma_max,
            "gamma_min": gamma_min,
            "device": device
        }
        super().__init__(params, defaults=defaults)

    
    def step(self):
        for group in self.param_groups:

            # betaとsigmaの更新
            beta = group["params"][0]
            gamma = group["params"][1]

            beta_update = beta.data - \
                group["lr_beta"] * beta.grad.data
            beta_update = max(beta_update, torch.tensor(group["beta_min"]))
            beta_update = min(beta_update, torch.tensor(group["beta_max"]))
            if not math.isnan(beta_update):
                #beta.data.copy_(torch.tensor(beta_update)
                beta.data.copy_(beta_update.clone().detach())

            gamma_update = gamma.data - \
                group["lr_gamma"] * gamma.grad.data
            gamma_update = max(gamma_update, torch.tensor(group["gamma_min"]))
            gamma_update = min(gamma_update, torch.tensor(group["gamma_max"]))
            if not math.isnan(gamma_update):
                #gamma.data.copy_(torch.tensor(gamma_update))
                gamma.data.copy_(gamma_update.clone().detach())

            # うめこみの更新
            for p in group["params"][2:]:
                # print("p.grad:", p.grad)
                if p.grad is None:
                    continue

                grad_norm = torch.norm(p.grad.data)
                grad_norm = torch.where(
                    grad_norm > 1, grad_norm, torch.tensor(1.0, device=p.device))
                h = (p.grad.data / grad_norm)
                update = p - group["lr_embeddings"] * h

                is_nan_inf = torch.isnan(update) | torch.isinf(update)
                update = torch.where(is_nan_inf, p, update)

                p.data.copy_(update)

class Euclidean(nn.Module):

    def __init__(
        self,
        n_nodes,
        n_dim,  # 次元より1つ多くデータを取る必要があることに注意
        k,
        R,
        init_range=0.01,
        sparse=True,
        device="cpu",
        calc_latent=True
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_dim = n_dim
        self.k = k
        self.R = R
        self.device = device
        self.calc_latent = calc_latent

    def latent_lik(
        self,
        x
    ):
        pass


    def forward(
        self,
        pairs,
        labels,
    ):
        # zを与えた下でのyの尤度
        loss = self.lik_y_given_z(
            pairs,
            labels
        )

        # z自体のロス
        # 座標を取得
        us = self.table(pairs[:, 0])
        vs = self.table(pairs[:, 1])

        if self.calc_latent:  # calc_latentがTrueの時のみ計算する
            lik_us = self.latent_lik(us, pairs[:, 0])
            lik_vs = self.latent_lik(vs, pairs[:, 1])
            loss = loss + (lik_us + lik_vs) / (self.n_nodes - 1)

        return loss
    
    def lik_y_given_z(
        self,
        pairs,
        labels
    ):
        pass

    def z(
        self
    ):
        z = self.table.weight.data
        lik_z = self.latent_lik(z).sum().item()
        return lik_z

    def get_euclidean_table(self):
        return self.table.weight.data.cpu().numpy()

    def calc_probability(
        self,
        samples,
    ):
        samples_ = torch.Tensor(samples).to(self.device).long()

        # 座標を取得
        us = self.table(samples_[:, 0])
        vs = self.table(samples_[:, 1])

        dist = e_dist(us, vs)
        p = torch.exp(-torch.logaddexp(torch.tensor([0.0]).to(
            self.device), self.beta * (dist - self.R)))
        print(p)

        return p.detach().cpu().numpy()

    def get_PC(
        self,
        sampling=True
    ):
        pass


class Gaussian(Euclidean):

    def __init__(
        self,
        n_nodes,
        n_dim,  # 次元より1つ多くデータを取る必要があることに注意
        k, #クラスタ数
        R,
        beta,
        gamma,
        eps_1,
        init_range=0.01,
        sparse=True,
        device="cpu",
        calc_latent=True
    ):
        super().__init__(
            n_nodes=n_nodes,
            n_dim=n_dim,  # 次元より1つ多くデータを取る必要があることに注意
            k=k,
            R=R,
            init_range=init_range,
            sparse=sparse,
            device=device,
            calc_latent=calc_latent
        )
        self.beta = nn.Parameter(torch.tensor(beta))
        self.gamma = nn.Parameter(torch.tensor(gamma))
        self.eps_1 = eps_1

        self.table = nn.Embedding(n_nodes, n_dim, sparse=sparse)
        nn.init.normal_(self.table.weight, 0, init_range)
        
        #追加したパラメータ
        self.k = k
        self.centroid = np.zeros((self.k, n_dim), dtype=np.float32)
        self.covariance_mat = np.zeros((self.k, n_dim, n_dim), dtype=np.float32)
        self.inv_covariance_mat = np.zeros((self.k, n_dim, n_dim), dtype=np.float32)
        self.pi = np.zeros((self.k,), dtype=np.float32)
        self.pi_each = np.zeros((n_nodes, self.k), dtype=np.float32)



    def latent_lik(
        self,
        x,
        node_index
    ):
        n_subnodes = x.shape[0]
        
        lik = torch.zeros(n_subnodes).to(self.device)
        
        covariance_mat = torch.from_numpy(self.covariance_mat.astype(np.float32)).clone()
        centroid = torch.from_numpy(self.centroid.astype(np.float32)).clone()
        pi = torch.from_numpy(self.pi.astype(np.float32)).clone()
        pi_each = torch.from_numpy(self.pi_each.astype(np.float32)).clone()
        
        for com in range(self.k):
            lik += pi_each[node_index, com] * (torch.ones(n_subnodes).to(self.device) * ((self.n_dim / 2) * torch.log(torch.tensor(2 * np.pi).to(self.device)) + 0.5 * torch.log(torch.det(covariance_mat[com]) + 0.000001)))
            Sigma_pinv = torch.linalg.pinv(covariance_mat[com])  # Pseudo-inverse
            lik_com = pi_each[node_index, com] * (- torch.log(pi[com] * torch.exp(-0.5 * torch.diag((x-centroid[com]).mm(Sigma_pinv.mm((x-centroid[com]).T)), 0))))
            lik_com[torch.isnan(lik_com)] = 0
            lik_com[torch.isinf(lik_com)] = 0
            lik += lik_com
        return lik

    def lik_y_given_z(
        self,
        pairs,
        labels
    ):
        # 座標を取得
        us = self.table(pairs[:, 0])
        vs = self.table(pairs[:, 1])

        # ロス計算
        dist = e_dist(us, vs)
        loss = torch.clone(labels).float()
        # 数値計算の問題をlogaddexpで回避
        # zを固定した下でのyのロス
        loss = torch.where(
            loss == 1,
            torch.logaddexp(torch.tensor([0.0]).to(
                self.device), self.beta * dist - self.gamma),
            torch.logaddexp(torch.tensor([0.0]).to(
                self.device), -self.beta * dist + self.gamma)
        )

        return loss

    def calc_dist(
        self,
        samples,
    ):
        samples_ = torch.Tensor(samples).to(self.device).long()

        # 座標を取得
        us = self.table(samples_[:, 0])
        vs = self.table(samples_[:, 1])

        dist = e_dist(us, vs)

        return dist.detach().cpu().numpy()

    def get_PC(
        self,
        beta_min,
        beta_max,
        gamma_min,
        gamma_max,
        sampling
    ):
        if sampling == False:
            # DNMLのPCの計算
            x_e = self.get_euclidean_table()
        else:
            idx = np.array(range(self.n_nodes))
            idx = np.random.permutation(
                idx)[:min(int(self.n_nodes * 0.1), 100)]
            x_e = self.get_euclidean_table()[idx, :]

        n_nodes_sample = len(x_e)
        #print(n_nodes_sample)

        #print(x_e)

        dist_mat = distance_mat(x_e, x_e)

        #print(dist_mat)

        is_nan_inf = np.isnan(dist_mat) | np.isinf(dist_mat)
        dist_mat = np.where(is_nan_inf, 2 * self.R, dist_mat)
        X = dist_mat
        # dist_mat
        # X = self.R - dist_mat
        for i in range(n_nodes_sample):
            X[i, i] = 0

        # I_n
        def sqrt_I_n(
            beta,
            gamma
        ):
            I_1_1 = np.sum(X**2 / ((np.cosh((beta * X - gamma) / 2.0) * 2)
                                   ** 2)) / (n_nodes_sample * (n_nodes_sample - 1))
            I_1_2 = np.sum(- X / ((np.cosh((beta * X - gamma) / 2.0) * 2)
                                  ** 2)) / (n_nodes_sample * (n_nodes_sample - 1))
            I_2_2 = 1 / ((np.cosh((beta * X - gamma) / 2.0) * 2)**2)
            for i in range(n_nodes_sample):
                I_2_2[i, i] = 0
            I_2_2 = np.sum(I_2_2) / (n_nodes_sample * (n_nodes_sample - 1))

            return np.sqrt(np.abs(I_1_1 * I_2_2 - I_1_2 * I_1_2))

        ret_1 = 0.5 * (np.log(self.n_nodes) + np.log(self.n_nodes - 1) - np.log(4 * np.pi)) + \
            np.log(integrate.dblquad(sqrt_I_n, gamma_min,
                                     gamma_max, beta_min, beta_max)[0])

        return ret_1


def calc_Euclidean(
    adj_mat,
    train_graph,
    positive_samples,
    negative_samples,
    lik_data,
    #x_lorentz,
    params_dataset,
    model_n_dim,
    model_k,
    burn_epochs,
    burn_batch_size,
    n_max_positives,
    n_max_negatives,
    lr_embeddings,
    lr_epoch_10,
    lr_beta,
    lr_gamma,
    beta_min,
    beta_max,
    gamma_min,
    gamma_max,
    eps_1,
    device,
    loader_workers=16,
    shuffle=True,
    sparse=False,
    calc_groundtruth=False
):

    print("model_n_dim:", model_n_dim)
    print("model_k:", model_k)
    

    print("pos data", len(positive_samples))
    print("neg data", len(negative_samples))
    print("len data", len(lik_data))

    # burn-inでの処理
    dataloader = DataLoader(
        NegGraph(train_graph, n_max_positives, n_max_negatives),
        shuffle=shuffle,
        batch_size=burn_batch_size,
        num_workers=loader_workers,
        pin_memory=True
    )
    
    # 真のデータ数
    n_data = params_dataset['n_nodes'] * (params_dataset['n_nodes'] - 1)
    

    # model
    model_gaussian = Gaussian(
        n_nodes=params_dataset['n_nodes'],
        n_dim=model_n_dim,  # モデルの次元
        k = model_k,
        R=params_dataset['R'],
        beta=1.0,
        gamma=params_dataset['R'],
        eps_1=eps_1,
        # init_range=0.001,
        init_range=10,
        sparse=sparse,
        device=device,
        calc_latent=True
    )

    # 最適化関数。
    sgd_gaussian = SGD_Gaussian(
        model_gaussian.parameters(),
        lr_embeddings=lr_embeddings,
        lr_beta=lr_beta,
        lr_gamma=lr_gamma,
        R=params_dataset['R'] * 2,
        beta_max=beta_max,
        beta_min=beta_min,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        device=device
    )

    model_gaussian.to(device)

    start = time.time()
    
    loss_x_z_best = np.inf
    loss_z_best = np.inf
    loss_best = np.inf
    model_best = model_gaussian
    num_fail = 0
    
    for epoch in range(burn_epochs):
        if epoch == 10:
            # batchサイズに対応して学習率変更
            sgd_gaussian.param_groups[0]["lr_embeddings"] = lr_epoch_10 * 1000
        
        if epoch % 10 == 0:
            if epoch == 0:
                g_mixture = mixture.GaussianMixture(n_components=model_gaussian.k,
                                                     #reg_covar=reg_covar,
                                                     #covariance_type='full',
                                                     #n_init=n_init
                                                   )
            else:
                g_mixture = mixture.GaussianMixture(n_components=model_gaussian.k, means_init=model_gaussian.centroid)
            
            g_mixture.fit(model_gaussian.table.weight.data)
            model_gaussian.centroid = g_mixture.means_.astype(np.float32)
            model_gaussian.covariance_mat = g_mixture.covariances_.astype(np.float32)
            model_gaussian.inv_covariance_mat = g_mixture.precisions_.astype(np.float32)
            model_gaussian.pi = g_mixture.weights_.astype(np.float32)
            model_gaussian.pi_each = g_mixture.predict_proba(model_gaussian.table.weight.data).astype(np.float32)
            
            loss_z = - g_mixture.score(model_gaussian.table.weight.data) * params_dataset['n_nodes']
            
            
            d_mat = distance_mat(model_gaussian.table.weight.data.numpy(), model_gaussian.table.weight.data.numpy())
            
            x_train = []
            y_train = []
            for i in range(params_dataset['n_nodes']):
                for j in range(i+1, params_dataset['n_nodes']):
                    if train_graph[i,j] != -1:
                        x_train.append(d_mat[i,j])
                        y_train.append(train_graph[i,j])
            x_train = np.array(x_train)
            y_train = np.array(y_train)

            # 対数尤度の計算
            loss_x_z = calculate_negative_log_likelihood(np.array([- model_gaussian.beta.item()]), model_gaussian.gamma.item(), x_train, y_train)


            print("loss:", loss_z + loss_x_z, "loss_z:", loss_z, "loss_x_z:", loss_x_z, "beta:", model_gaussian.beta.item(), "gamma:", model_gaussian.gamma.item())
            
            if loss_x_z + loss_z < loss_best:
                loss_best = loss_x_z + loss_z
                model_best = model_gaussian
                g_mixture_best = g_mixture
                loss_x_z_best = loss_x_z
                loss_z_best = loss_z
                num_fail = 0
            else:
                num_fail += 1
                if num_fail >= 10:
                    print("final_epoch:", epoch)
                    break

        losses_gaussian = []
        
        for pairs, labels in dataloader:
            pairs = pairs.reshape((-1, 2))
            labels = labels.reshape(-1)

            pairs = pairs.to(device)
            labels = labels.to(device)

            # DNML-Gaussian
            sgd_gaussian.zero_grad()
            loss_gaussian = model_gaussian(pairs, labels).mean()
            loss_gaussian.backward()
            sgd_gaussian.step()
            losses_gaussian.append(loss_gaussian)

        print("epoch:", epoch, "loss_gaussian:",
              torch.Tensor(losses_gaussian).mean().item())

    

    g_mixture = mixture.GaussianMixture(n_components=model_gaussian.k, means_init=model_gaussian.centroid)

    g_mixture.fit(model_gaussian.table.weight.data)
    model_gaussian.centroid = g_mixture.means_.astype(np.float32)
    model_gaussian.covariance_mat = g_mixture.covariances_.astype(np.float32)
    model_gaussian.inv_covariance_mat = g_mixture.precisions_.astype(np.float32)
    model_gaussian.pi = g_mixture.weights_.astype(np.float32)
    model_gaussian.pi_each = g_mixture.predict_proba(model_gaussian.table.weight.data).astype(np.float32)

    loss_z = - g_mixture.score(model_gaussian.table.weight.data) * params_dataset['n_nodes']


    d_mat = distance_mat(model_gaussian.table.weight.data.numpy(), model_gaussian.table.weight.data.numpy())

    x_train = []
    y_train = []
    for i in range(params_dataset['n_nodes']):
        for j in range(i+1, params_dataset['n_nodes']):
            if train_graph[i,j] != -1:
                x_train.append(d_mat[i,j])
                y_train.append(train_graph[i,j])
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    # 対数尤度の計算
    loss_x_z = calculate_negative_log_likelihood(np.array([- model_gaussian.beta.item()]), model_gaussian.gamma.item(), x_train, y_train)


    print("loss:", loss_z + loss_x_z, "loss_z:", loss_z, "loss_x_z:", loss_x_z, "beta:", model_gaussian.beta.item(), "gamma:", model_gaussian.gamma.item())

    if loss_x_z + loss_z < loss_best:
        loss_best = loss_x_z + loss_z
        model_best = model_gaussian
        g_mixture_best = g_mixture
        loss_x_z_best = loss_x_z
        loss_z_best = loss_z
        num_fail = 0
    
    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

    
    
    d_mat = distance_mat(model_best.table.weight.data.numpy(), model_best.table.weight.data.numpy())
            
    x_train = []
    y_train = []
    for i in range(params_dataset['n_nodes']):
        for j in range(i+1, params_dataset['n_nodes']):
            if train_graph[i,j] != -1:
                x_train.append(d_mat[i,j])
                y_train.append(train_graph[i,j])
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    # 現在のモデルのパラメータを初期値として新しいデータに対して再トレーニング
    model_lr = LogisticRegression(warm_start=True)
    model_lr.coef_ = np.array([[- model_best.beta.item()]])
    model_lr.intercept_ = np.array([model_best.gamma.item()])

    # 新しいデータでトレーニング
    model_lr.fit(x_train.reshape(-1, 1), y_train)
    
    print("beta:", - model_lr.coef_[0,0], "gamma:", model_lr.intercept_[0])


    loss_x_z = -np.sum(y_train * np.log(model_lr.predict_proba(x_train.reshape(-1, 1))[:, 1]) + (1 - y_train) * np.log(model_lr.predict_proba(x_train.reshape(-1, 1))[:, 0])) * n_data / len(x_train) / 2

    loss_x_z_best = loss_x_z
    
    
    loss_best = loss_x_z_best + loss_z_best
    
    # 尤度計算
    latent_z = g_mixture_best.predict(model_best.table.weight.data)
    
    nk = np.bincount(latent_z, minlength=model_k)

    if min(nk) <= 0:
        basescore_z = np.nan
        basescore_phi_given_z = np.nan
    else:
        pi = g_mixture_best.weights_
        covariances = g_mixture_best.covariances_
        basescore_z = 0
        basescore_phi_given_z = 0
        for k in range(model_k):
            basescore_z -= nk[k] * np.log(pi[k])
            basescore_phi_given_z += 0.5 * nk[k] * model_n_dim * np.log(2 * math.pi * math.e)
            basescore_phi_given_z += 0.5 * nk[k] * np.log(det(covariances[k]))
    
    # create log_pc_array
    log_pc_array = log_pc_gmm(K_max=model_k, N_max=params_dataset['n_nodes'], D=model_n_dim)

    pc_z = np.log(_pc_multinomial(params_dataset['n_nodes'], model_k))
    
    pc_phi_given_z = 0
    for k in range(model_k):
        Z_k = sum(latent_z == k)
        pc_phi_given_z += log_pc_array[1, Z_k]


    basescore_y_given_phi = loss_x_z_best
    
    basescore_y_and_phi_and_z = basescore_y_given_phi + basescore_phi_given_z + basescore_z


    pc_y_given_phi = model_best.get_PC(
        beta_min,
        beta_max,
        gamma_min,
        gamma_max,
        sampling=True
    )

    DNML_Gaussian = basescore_y_and_phi_and_z + \
        pc_y_given_phi + pc_phi_given_z + pc_z
    AIC_Gaussian = basescore_y_and_phi_and_z + \
        2 + model_n_dim * (model_n_dim + 3) / 2 + model_k - 1
    BIC_Gaussian = basescore_y_and_phi_and_z + \
        np.log(params_dataset['n_nodes']) + np.log(params_dataset['n_nodes'] - 1) - np.log(2) + \
            (model_n_dim * (model_n_dim + 3) / 4 + (model_k - 1) / 2) * np.log(params_dataset['n_nodes'])

    # Calculate AUC from probability
    def calc_AUC_from_prob(
        positive_dist,
        negative_dist
    ):

        pred = np.append(-positive_dist, -negative_dist)
        ground_truth = np.append(np.ones(len(positive_dist)),
                                 np.zeros(len(negative_dist)))
        AUC = metrics.roc_auc_score(ground_truth, pred)
        return AUC

    # latentを計算したものでのAUC
    AUC_Gaussian = calc_AUC_from_prob(
        model_best.calc_dist(positive_samples),
        model_best.calc_dist(negative_samples)
    )
    
    # model_kとは別で，phiをk=1,2,...,10でクラスタリングしたときの，DNML_phi_and_zを計算
    DNML_phi_and_z = {}
    for k_gmm in [1,2,3,4,5,6,7,8,9,10]:
        dnml_k_gmm = np.inf
        for _ in range(10):
            g_mixture = mixture.GaussianMixture(n_components=k_gmm)
            g_mixture.fit(model_best.table.weight.data)
            latent_z = g_mixture.predict(model_best.table.weight.data)

            nk = np.bincount(latent_z, minlength=k_gmm)

            if min(nk) <= 0:
                basescore_z_k_gmm = np.nan
                basescore_phi_given_z_k_gmm = np.nan
            else:
                pi = g_mixture.weights_
                covariances = g_mixture.covariances_
                basescore_z_k_gmm = 0
                basescore_phi_given_z_k_gmm = 0
                for i in range(k_gmm):
                    basescore_z_k_gmm -= nk[i] * np.log(pi[i])
                    basescore_phi_given_z_k_gmm += 0.5 * nk[i] * model_n_dim * np.log(2 * math.pi * math.e)
                    basescore_phi_given_z_k_gmm += 0.5 * nk[i] * np.log(det(covariances[i]))

            # create log_pc_array
            log_pc_array = log_pc_gmm(K_max=k_gmm, N_max=params_dataset['n_nodes'], D=model_n_dim)

            pc_z_k_gmm = np.log(_pc_multinomial(params_dataset['n_nodes'], k_gmm))

            pc_phi_given_z_k_gmm = 0
            for i in range(k_gmm):
                Z_k = sum(latent_z == i)
                pc_phi_given_z_k_gmm += log_pc_array[1, Z_k]
            
            dnml_temp = basescore_phi_given_z_k_gmm + basescore_z_k_gmm + pc_phi_given_z_k_gmm + pc_z_k_gmm
            if dnml_temp < dnml_k_gmm:
                dnml_k_gmm = dnml_temp
        
        DNML_phi_and_z[k_gmm] = dnml_k_gmm


    print("-log p(y, phi, z):", basescore_y_and_phi_and_z)
    print("-log p(y|phi):", basescore_y_given_phi)
    print("-log p(phi|z):", basescore_phi_given_z)
    print("-log p(z):", basescore_z)
    print("pc_y_given_phi", pc_y_given_phi)
    print("pc_phi_given_z", pc_phi_given_z)
    print("pc_z", pc_z)
    print("DNML-Gaussian:", DNML_Gaussian)
    print("AIC_Gaussian:", AIC_Gaussian)
    print("BIC_Gaussian:", BIC_Gaussian)
    print("AUC_Gaussian:", AUC_Gaussian)

    ret = {
        "DNML_Gaussian": DNML_Gaussian,
        "AIC_Gaussian": AIC_Gaussian,
        "BIC_Gaussian": BIC_Gaussian,
        "AUC_Gaussian": AUC_Gaussian,
        "-log p(y, phi, z)": basescore_y_and_phi_and_z,
        "-log p(y|phi)": basescore_y_given_phi,
        "-log p(phi|z)": basescore_phi_given_z,
        "-log p(z)": basescore_z,
        "pc_y_given_phi": pc_y_given_phi,
        "pc_phi_given_z": pc_phi_given_z,
        "pc_z": pc_z,
        "model_gaussian": model_best,
        "DNML_phi_and_z_1": DNML_phi_and_z[1],
        "DNML_phi_and_z_2": DNML_phi_and_z[2],
        "DNML_phi_and_z_3": DNML_phi_and_z[3],
        "DNML_phi_and_z_4": DNML_phi_and_z[4],
        "DNML_phi_and_z_5": DNML_phi_and_z[5],
        "DNML_phi_and_z_6": DNML_phi_and_z[6],
        "DNML_phi_and_z_7": DNML_phi_and_z[7],
        "DNML_phi_and_z_8": DNML_phi_and_z[8],
        "DNML_phi_and_z_9": DNML_phi_and_z[9],
        "DNML_phi_and_z_10": DNML_phi_and_z[10]
    }

    return ret



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='sbm')
    parser.add_argument('n_nodes', help='n_nodes')
    parser.add_argument('n_block', help='n_block')
    parser.add_argument('p_other', help='p_other')
    parser.add_argument('p_dif', help='p_dif')
    args = parser.parse_args()
    #print(args)


    if args.n_nodes == "0":
        n_nodes = 400
    elif args.n_nodes == "1":
        n_nodes = 800
    elif args.n_nodes == "2":
        n_nodes = 1600
    elif args.n_nodes == "3":
        n_nodes = 3200
    elif args.n_nodes == "4":
        n_nodes = 6400
    
    n_block = int(args.n_block)
    
    if args.p_other == "0":
        p_other = 0.1
    elif args.p_other == "1":
        p_other = 0.2
    elif args.p_other == "2":
        p_other = 0.3
    
    if args.p_dif == "0":
        p_same = p_other + 0.1
    elif args.p_dif == "1":
        p_same = p_other + 0.2
    elif args.p_dif == "2":
        p_same = p_other + 0.3
    elif args.p_dif == "3":
        p_same = p_other + 0.4

    
    
    
    params_dataset = {
        'n_nodes': n_nodes,
        'n_dim': 8, #仮
        'R': np.log(n_nodes),
        'beta': 0.2 #仮
    }

    
    
    adj_mat = np.zeros((n_nodes, n_nodes))
    adj_mat[:,:] = p_other
    for i in range(n_block):
        adj_mat[i*n_nodes//n_block:(i+1)*n_nodes//n_block, i*n_nodes//n_block:(i+1)*n_nodes//n_block] = p_same
    for i in range(n_block):
        adj_mat[i, i] = 1

    # サンプリング用の行列
    sampling_mat = np.random.uniform(0, 1, adj_mat.shape)
    sampling_mat = np.triu(
        sampling_mat) + np.triu(sampling_mat).T - np.diag(sampling_mat.diagonal())

    adj_mat = np.where(sampling_mat < adj_mat, 1, 0)
    
    
    
    # パラメータ
    burn_epochs = 1000
    burn_batch_size = min(int(params_dataset["n_nodes"] * 0.2), 50)
    n_max_positives = min(int(params_dataset["n_nodes"] * 0.02), 10)
    lr_beta = 0.001
    lr_gamma = 0.001
    beta_min = 0.1
    beta_max = 10.0
    gamma_min = 0.1
    gamma_max = 10.0
    eps_1 = 1e-6
    # それ以外
    loader_workers = 0 #8
    print("loader_workers: ", loader_workers)
    shuffle = True
    sparse = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device:',device)

    positive_samples, negative_samples, train_graph, lik_data = create_test_for_link_prediction(
        adj_mat=adj_mat,
        params_dataset=params_dataset
    )

    # negative samplingの比率を平均次数から決定
    pos_train_graph = len(np.where(train_graph == 1)[0])
    neg_train_graph = len(np.where(train_graph == 0)[0])
    ratio = neg_train_graph / pos_train_graph
    print("ratio:", ratio)

    ratio = min(ratio, 10)

    n_max_negatives = int(n_max_positives * ratio)
    print("n_max_negatives:", n_max_negatives)
    lr_embeddings = 0.1
    lr_epoch_10 = 10.0 * \
        (burn_batch_size * (n_max_positives + n_max_negatives)) / \
        32 / 100  # batchサイズに対応して学習率変更
    
    
    dnml_k_1 = {}
    
    model_k = 1
    for model_n_dim in [2,4,8]:
        ret = calc_Euclidean(
            adj_mat=adj_mat,
            train_graph=train_graph,
            positive_samples=positive_samples,
            negative_samples=negative_samples,
            lik_data=lik_data,
            params_dataset=params_dataset,
            model_n_dim=model_n_dim,
            model_k=model_k,
            burn_epochs=burn_epochs,
            burn_batch_size=burn_batch_size,
            n_max_positives=n_max_positives,
            n_max_negatives=n_max_negatives,
            lr_embeddings=lr_embeddings,
            lr_epoch_10=lr_epoch_10,
            lr_beta=lr_beta,
            lr_gamma=lr_gamma,
            beta_min=beta_min,
            beta_max=beta_max,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            eps_1=eps_1,
            device=device,
            loader_workers=loader_workers,
            shuffle=shuffle,
            sparse=sparse,
            calc_groundtruth=False
        )
        
        dnml_k_1[model_n_dim] = ret["DNML_Gaussian"]


        ret.pop('model_gaussian')

        ret["model_n_dims"] = model_n_dim
        ret["model_k"] = model_k #追加
        ret["n_nodes"] = params_dataset["n_nodes"]
        ret["n_block"] = n_block
        ret["p_other"] = p_other
        ret["p_same"] = p_same
        ret["R"] = params_dataset["R"]
        ret["burn_epochs"] = burn_epochs
        ret["burn_batch_size"] = burn_batch_size
        ret["n_max_positives"] = n_max_positives
        ret["n_max_negatives"] = n_max_negatives
        ret["lr_embeddings"] = lr_embeddings
        ret["lr_epoch_10"] = lr_epoch_10
        ret["lr_beta"] = lr_beta
        ret["lr_gamma"] = lr_gamma
        ret["beta_max"] = beta_max
        ret["beta_min"] = beta_min
        ret["gamma_max"] = gamma_max
        ret["gamma_min"] = gamma_min
        ret["eps_1"] = eps_1

        row = pd.DataFrame(ret.values(), index=ret.keys()).T

        row = row.reindex(columns=[
            "model_n_dims",
            "model_k",
            "n_nodes",
            "n_block",
            "p_other",
            "p_same",
            "R",
            "DNML_Gaussian",
            "AIC_Gaussian",
            "BIC_Gaussian",
            "AUC_Gaussian",
            "-log p(y, phi, z)",
            "-log p(y|phi)",
            "-log p(phi|z)",
            "-log p(z)",
            "pc_y_given_phi",
            "pc_phi_given_z",
            "pc_z",
            "DNML_phi_and_z_1",
            "DNML_phi_and_z_2",
            "DNML_phi_and_z_3",
            "DNML_phi_and_z_4",
            "DNML_phi_and_z_5",
            "DNML_phi_and_z_6",
            "DNML_phi_and_z_7",
            "DNML_phi_and_z_8",
            "DNML_phi_and_z_9",
            "DNML_phi_and_z_10",
            "burn_epochs",
            "n_max_positives",
            "n_max_negatives",
            "lr_embeddings",
            "lr_epoch_10",
            "lr_beta",
            "lr_gamma",
            "beta_max",
            "beta_min",
            "gamma_max",
            "gamma_min",
            "eps_1"
        ]
        )

        filepath = "result_artificial_sbm_1.csv"

        if os.path.exists(filepath):
            result_previous = pd.read_csv(filepath)
            result = pd.concat([result_previous, row])
            result.to_csv(filepath, index=False)
        else:
            row.to_csv(filepath, index=False)
    
    
    model_n_dim = min(dnml_k_1, key=dnml_k_1.get)
    for model_k in [2,4,8]:
        ret = calc_Euclidean(
            adj_mat=adj_mat,
            train_graph=train_graph,
            positive_samples=positive_samples,
            negative_samples=negative_samples,
            lik_data=lik_data,
            params_dataset=params_dataset,
            model_n_dim=model_n_dim,
            model_k=model_k,
            burn_epochs=burn_epochs,
            burn_batch_size=burn_batch_size,
            n_max_positives=n_max_positives,
            n_max_negatives=n_max_negatives,
            lr_embeddings=lr_embeddings,
            lr_epoch_10=lr_epoch_10,
            lr_beta=lr_beta,
            lr_gamma=lr_gamma,
            beta_min=beta_min,
            beta_max=beta_max,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            eps_1=eps_1,
            device=device,
            loader_workers=loader_workers,
            shuffle=shuffle,
            sparse=sparse,
            calc_groundtruth=False
        )


        ret.pop('model_gaussian')

        ret["model_n_dims"] = model_n_dim
        ret["model_k"] = model_k #追加
        ret["n_nodes"] = params_dataset["n_nodes"]
        ret["n_block"] = n_block
        ret["p_other"] = p_other
        ret["p_same"] = p_same
        ret["R"] = params_dataset["R"]
        ret["burn_epochs"] = burn_epochs
        ret["burn_batch_size"] = burn_batch_size
        ret["n_max_positives"] = n_max_positives
        ret["n_max_negatives"] = n_max_negatives
        ret["lr_embeddings"] = lr_embeddings
        ret["lr_epoch_10"] = lr_epoch_10
        ret["lr_beta"] = lr_beta
        ret["lr_gamma"] = lr_gamma
        ret["beta_max"] = beta_max
        ret["beta_min"] = beta_min
        ret["gamma_max"] = gamma_max
        ret["gamma_min"] = gamma_min
        ret["eps_1"] = eps_1

        row = pd.DataFrame(ret.values(), index=ret.keys()).T

        row = row.reindex(columns=[
            "model_n_dims",
            "model_k",
            "n_nodes",
            "n_block",
            "p_other",
            "p_same",
            "R",
            "DNML_Gaussian",
            "AIC_Gaussian",
            "BIC_Gaussian",
            "AUC_Gaussian",
            "-log p(y, phi, z)",
            "-log p(y|phi)",
            "-log p(phi|z)",
            "-log p(z)",
            "pc_y_given_phi",
            "pc_phi_given_z",
            "pc_z",
            "DNML_phi_and_z_1",
            "DNML_phi_and_z_2",
            "DNML_phi_and_z_3",
            "DNML_phi_and_z_4",
            "DNML_phi_and_z_5",
            "DNML_phi_and_z_6",
            "DNML_phi_and_z_7",
            "DNML_phi_and_z_8",
            "DNML_phi_and_z_9",
            "DNML_phi_and_z_10",
            "burn_epochs",
            "n_max_positives",
            "n_max_negatives",
            "lr_embeddings",
            "lr_epoch_10",
            "lr_beta",
            "lr_gamma",
            "beta_max",
            "beta_min",
            "gamma_max",
            "gamma_min",
            "eps_1"
        ]
        )

        filepath = "result_artificial_sbm_1.csv"

        if os.path.exists(filepath):
            result_previous = pd.read_csv(filepath)
            result = pd.concat([result_previous, row])
            result.to_csv(filepath, index=False)
        else:
            row.to_csv(filepath, index=False)
