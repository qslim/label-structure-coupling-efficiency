import random
import sys
import argparse
import yaml
import os
import math
import time
import pickle as pkl
import scipy as sp
from scipy import io
import numpy as np
from ogb.nodeproppred.dataset_dgl import DglNodePropPredDataset
import pandas as pd
import networkx as nx
from sklearn.preprocessing import label_binarize
import dgl
import torch
from numpy.linalg import eig, eigh
from utils import seed_everything


def normalize_graph(g, power=-0.5, norm_type='laplacian'):
    g = np.array(g)
    g = g + g.T
    g[g > 0.] = 1.0
    deg = g.sum(axis=1).reshape(-1)
    deg[deg == 0.] = 1.0
    deg = np.diag(deg ** power)
    adj = np.dot(np.dot(deg, g), deg)
    if norm_type == 'laplacian':
        if power == -0.5:
            _deg = np.eye(g.shape[0])
        else:
            _deg = adj.sum(axis=1).reshape(-1)
            # _deg[_deg == 0.] = 1.0
            _deg = np.diag(_deg)
        res = _deg - adj
    elif norm_type == 'adjacency':
        res = adj
    else:
        raise NotImplementedError
    return res


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def feature_normalize(x):
    x = np.array(x)
    rowsum = x.sum(axis=1, keepdims=True)
    rowsum = np.clip(rowsum, 1, 1e10)
    return x / rowsum


def load_data(dataset_str):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("node_raw_data/{}/ind.{}.{}".format(dataset_str, dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("node_raw_data/{}/ind.{}.test.index".format(dataset_str, dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.sparse.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.sparse.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    return adj, features, labels


def compute_feat_graph(x, dataset, pair_dis_type='cosine'):
    x = (x - x.mean(axis=1, keepdims=True)) / np.maximum(x.std(axis=1, keepdims=True), 1e-8)
    # x = x - x.mean(axis=1, keepdims=True)
    # x = x - x.mean(axis=0, keepdims=True)
    if pair_dis_type == 'cosine':
        # Step 1: Normalize each row vector to have unit length
        norm_x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
        # Step 2: Compute the cosine similarity matrix
        if dataset == 'cora':
            feat_graph = np.dot(norm_x, norm_x.T)
        else:
            feat_graph = np.absolute(np.dot(norm_x, norm_x.T))
            # feat_graph = np.dot(norm_x, norm_x.T) + 1.0
        # print(feat_similarity)
    elif pair_dis_type == 'euclidean':
        n = x.shape[0]
        feat_graph = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                dist = np.linalg.norm(x[i] - x[j])
                feat_graph[i, j] = dist
                feat_graph[j, i] = dist
    else:
        raise NotImplementedError

    return feat_graph


def generate_node_data(dataset, config):
    
    if dataset in ['cora', 'citeseer', 'pubmed']:

        adj, x, y = load_data(dataset)
        adj = adj.todense()
        x = x.todense()
        x = feature_normalize(x)

    elif dataset in ['photo', 'computers']:
        data = np.load('node_raw_data/amazon_electronics_' + dataset + '.npz', allow_pickle=True)
        adj = sp.sparse.csr_matrix((data['adj_data'], data['adj_indices'], data['adj_indptr']), 
                            shape=data['adj_shape']).toarray()
        feat = sp.sparse.csr_matrix((data['attr_data'], data['attr_indices'], data['attr_indptr']), 
                            shape=data['attr_shape']).toarray()
        x = feature_normalize(feat)
        y = data['labels']

    elif dataset in ['chameleon', 'squirrel', 'actor', 'cornell', 'texas']:
        edge_df = pd.read_csv('node_raw_data/{}/'.format(dataset) + 'out1_graph_edges.txt', sep='\t')
        node_df = pd.read_csv('node_raw_data/{}/'.format(dataset) + 'out1_node_feature_label.txt', sep='\t')
        feature = node_df[node_df.columns[1]]
        y = node_df[node_df.columns[2]]

        num_nodes = len(y)
        adj = np.zeros((num_nodes, num_nodes))

        source = list(edge_df[edge_df.columns[0]])
        target = list(edge_df[edge_df.columns[1]])

        for i in range(len(source)):
            adj[source[i], target[i]] = 1.
            adj[target[i], source[i]] = 1.
    
        if dataset == 'actor':
            # for sparse features
            nfeat = 932
            x = np.zeros((len(y), nfeat))

            feature = list(feature)
            feature = [feat.split(',') for feat in feature]
            for ind, feat in enumerate(feature):
                for ff in feat:
                    x[ind, int(ff)] = 1.

            x = feature_normalize(x)
        else:
            feature = list(feature)
            feature = [feat.split(',') for feat in feature]
            new_feat = []

            for feat in feature:
                new_feat.append([int(f) for f in feat])
            x = np.array(new_feat)
            x = feature_normalize(x)

    else:
        raise NotImplementedError

    tic = time.time()
    e, u = [], []
    for pow in config['norm_power']:
        _e, _u = eigh(normalize_graph(adj, power=pow, norm_type=config['graph_norm_type']))
        e.append(_e)
        u.append(_u)
    e, u = np.concatenate(e, axis=0), np.concatenate(u, axis=1)

    if config['pair_trunc'] != 0:
        feat_graph = compute_feat_graph(x, dataset=dataset, pair_dis_type=config['pair_dis_type'])
        e_feat, u_feat = eigh(normalize_graph(feat_graph, norm_type=config['pair_norm_type']))
        pair_trunc = config['pair_trunc']
        if pair_trunc == 'all':
            pass
        elif pair_trunc < 0:
            e_feat, u_feat = e_feat[pair_trunc:], u_feat[:, pair_trunc:]
        else:
            e_feat, u_feat = e_feat[:pair_trunc], u_feat[:, :pair_trunc]
        e, u = np.concatenate((e, e_feat), axis=0), np.concatenate((u, u_feat), axis=1)
    print(time.time() - tic)

    e = torch.FloatTensor(e)
    u = torch.FloatTensor(u)
    x = torch.FloatTensor(x)
    y = torch.LongTensor(y)

    torch.save([e, u, x, y], 'data/{}.pt'.format(dataset))
    print(e.shape)
    print(u.shape)


def generate_node_data_ogbn(dataset, config):
    def normalize_graph(g, power=-0.5, norm_type='laplacian'):
        # adj = g.adj(scipy_fmt='csr')
        adj = g.adj_external(scipy_fmt='csr')
        deg = np.array(adj.sum(axis=0)).flatten()
        deg[deg == 0.] = 1.0
        deg = sp.sparse.diags(deg ** power)
        adj = deg.dot(adj.dot(deg))
        if norm_type == 'laplacian':
            if power == -0.5:
                deg = sp.sparse.eye(g.num_nodes())
            else:
                deg = np.array(adj.sum(axis=0)).flatten()
                deg = sp.sparse.diags(deg)
            res = deg - adj
        elif norm_type == 'adjacency':
            res = adj
        else:
            raise NotImplementedError
        return res

    def load_fb100_dataset():
        mat = io.loadmat('node_raw_data/Penn94.mat')
        A = mat['A']
        metadata = mat['local_info']

        edge_index = A.nonzero()
        metadata = metadata.astype(int)
        label = metadata[:, 1] - 1  # gender label, -1 means unlabeled

        # make features into one-hot encodings
        feature_vals = np.hstack((np.expand_dims(metadata[:, 0], 1), metadata[:, 2:]))
        features = np.empty((A.shape[0], 0))
        for col in range(feature_vals.shape[1]):
            feat_col = feature_vals[:, col]
            feat_onehot = label_binarize(feat_col, classes=np.unique(feat_col))
            features = np.hstack((features, feat_onehot))

        node_feat = torch.tensor(features, dtype=torch.float)
        num_nodes = metadata.shape[0]
        label = torch.LongTensor(label)

        g = dgl.graph((edge_index[0], edge_index[1]), num_nodes=num_nodes)

        return g, node_feat, label

    if dataset in ['arxiv', 'products']:
        data = DglNodePropPredDataset('ogbn-' + dataset)
        g = data[0][0]
        x = g.ndata['feat']
        y = data[0][1]
    elif dataset == 'penn':
        g, x, y = load_fb100_dataset()
    else:
        raise NotImplementedError
    g = dgl.add_reverse_edges(g)
    g = dgl.to_simple(g)
    e, u = [], []
    tic = time.time()
    trunc_k = 100
    lm_or_sm = 'LM'
    for pow in config['norm_power']:
        norm_g = normalize_graph(g, power=pow, norm_type=config['graph_norm_type'])
        # _e, _u = sp.sparse.linalg.eigsh(norm_g, k=trunc_k, which='SM', tol=1e-5)
        # e.append(_e)
        # u.append(_u)
        _e, _u = sp.sparse.linalg.eigsh(norm_g, k=trunc_k, which=lm_or_sm, tol=1e-5)
        e.append(_e)
        u.append(_u)
    print(time.time() - tic)
    e, u = np.concatenate(e, axis=0), np.concatenate(u, axis=1)

    e = torch.FloatTensor(e)
    u = torch.FloatTensor(u)

    torch.save([e, u], 'data/' + dataset + '_' + str(config['graph_norm_type']) + str(config['norm_power']) + '_' + lm_or_sm + str(trunc_k) + '.pt')
    torch.save([x, y], 'data/' + dataset + '_feature_label.pt')
    print(e.shape)
    print(u.shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--dataset', default='none')

    args = parser.parse_args()

    seed_everything(args.seed)

    config = yaml.load(open('config.yaml'), Loader=yaml.SafeLoader)[args.dataset]

    if args.dataset in ['arxiv', 'products']:
        generate_node_data_ogbn(args.dataset, config)
    else:
        generate_node_data(args.dataset, config)
