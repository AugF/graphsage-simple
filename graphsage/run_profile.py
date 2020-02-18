import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import time
import random
from sklearn.metrics import f1_score

import pickle as pkl
import scipy.sparse as sp
import sys

from graphsage.encoders import Encoder
from graphsage.aggregators import MeanAggregator
from graphsage.model import SupervisedGraphSage


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_data(dataset_str):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    for i in graph:
        graph[i] = set(graph[i])

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    labels = np.argmax(labels, 1)
    return features.todense(), labels, graph

def run_dataset(dataset_str, num_nodes, feature_dims, classes, hidden_dims,
               n_train, n_val, n_test,
               batch_iters = 200, batch_size = 1024, manual_seed = 1,
               gcn_flag = True, cuda_flag = False):

    np.random.seed(manual_seed)
    random.seed(manual_seed)

    feat_data, labels, adj_lists = load_data(dataset_str)
    features = nn.Embedding(num_nodes, feature_dims)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
   # features.cuda()

    agg1 = MeanAggregator(features, cuda=True)
    enc1 = Encoder(features, feature_dims, hidden_dims, adj_lists, agg1, gcn=True, cuda=False)
    agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), cuda=False)
    enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, hidden_dims, adj_lists, agg2,
            base_model=enc1, gcn=True, cuda=False)
    enc1.num_samples = num_samples[0]
    enc2.num_samples = num_samples[1]

    graphsage = SupervisedGraphSage(classes, enc2)
#    graphsage.cuda()
    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[: n_test]
    val = rand_indices[n_val: n_train]
    train = list(rand_indices[n_train:])

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, graphsage.parameters()), lr=0.7)
    times = []
    for batch in range(batch_iters):
        batch_nodes = train[:batch_size]
        random.shuffle(train)
        start_time = time.time()
        optimizer.zero_grad()
        loss = graphsage.loss(batch_nodes,
                Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time-start_time)
        print(batch, loss.item())

    val_output = graphsage.forward(val)
    print("Validation F1:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro"))
    print("Average batch time:", np.mean(times))


def run_cora():
    dataset_str = "cora"
    num_nodes, feature_dims, classes, hidden_dims = 2708, 1433, 7, 128
    n_train, n_val, n_test = 1500, 1000, 1000  # val: [n_val: n_train], train: [n_train:], test: [n_test:,]
    num_samples = [5, 5]
    batch_iters, batch_size = 100, 256
    run_dataset(dataset_str, num_nodes, feature_dims, classes, hidden_dims,
                n_train, n_val, n_test, batch_size=batch_size, batch_iters=batch_iters)

def run_pubmed():
    dataset_str = "pubmed"
    num_nodes, feature_dims, classes, hidden_dims = 19717, 500, 3, 128
    n_train, n_val, n_test = 1500, 1000, 1000  # val: [n_val: n_train], train: [n_train:], test: [n_test:,]
    num_samples = [10, 25]
    batch_iters, batch_size = 200, 1024
    run_dataset(dataset_str, num_nodes, feature_dims, classes, hidden_dims,
                n_train, n_val, n_test, batch_size=batch_size, batch_iters=batch_iters)

if __name__ == '__main__':
    run_cora()
    # pubmed
