# -*- coding: utf-8 -*-
# @Time    : 2020/4/15 19:40
# @Author  : Aurora
# @File    : data_loader_new.py
# @Function: 

import numpy as np
import torch
import torch.utils.data
from sklearn.model_selection import train_test_split

def getGraphFromFile(path_edges, num_u, num_v):
    ratings = open(path_edges, 'r', encoding='utf-8')
    G = np.ones([num_u, num_v], dtype=int)
    G = -G
    for line in ratings:
        u, v, label = line.strip().split(';')
        u, v, label = int(u), int(v), int(label)
        G[u, v] = label
    return G

def getGraphFromArray(edges, labels, num_u, num_v):
    G = np.ones([num_u, num_v], dtype=int)
    G = -G
    for i, edge in enumerate(edges):
        u, v = edge
        u, v, label = int(u), int(v), int(labels[i])
        G[u, v] = label
    return G

def get_train_test_withcount(path_edges, test_size, count):
    ratings = open(path_edges, 'r', encoding='utf-8')
    uv_list = []
    label_list = []
    for line in ratings:
        u, v, label = line.strip().split(';')
        uv_list.append([int(u), int(v)])
        label_list.append(int(label))
        count -= 1
        if count <=0:
            break

    uv_list = np.array(uv_list, dtype=int)
    label_list = np.array(label_list, dtype=int)
    train_uv_list, test_uv_list, train_label_list, test_label_list = train_test_split(uv_list, label_list,test_size=test_size, random_state=0)
    print('trainset and testset splited.')
    print('train samples: {0}; test samples: {1}\n'.format(len(train_label_list), len(test_label_list)))
    return train_uv_list, test_uv_list, train_label_list, test_label_list


def getL1paths(graph, uv):
    num_u, num_v = graph.shape
    nodes_L1paths = []
    print('num_u, num_v:', graph.shape)
    if uv==True:
        for u in range(num_u):
            connection = graph[u, :]
            L1neighbors = np.where(connection!=-1)[0] # array([vid, vid, vid, ...])
            num_neighbors = len(L1neighbors)
            # if num_neighbors == 0:
            #     nodes_L1paths.append()
            #     continue
            L1paths = np.empty([num_neighbors,2], dtype=int)
            for i, neighbor in enumerate(L1neighbors):
                L1paths[i, 0] = connection[neighbor]
                L1paths[i, 1] = neighbor
            nodes_L1paths.append(L1paths)
    else:
        for v in range(num_v):
            connection = graph[:, v]
            L1neighbors = np.where(connection!=-1)[0] # array([uid, uid, uid, ...])
            num_neighbors = len(L1neighbors)
            L1paths = np.empty([num_neighbors,2], dtype=int)

            for i, neighbor in enumerate(L1neighbors):
                L1paths[i, 0] = connection[neighbor]
                L1paths[i, 1] = neighbor
            nodes_L1paths.append(L1paths)
    return nodes_L1paths

def getL2paths(graph, u_L1paths, v_L1paths, sample_rate, uv):
    num_u, num_v = graph.shape
    nodes_L2paths = []
    print('num_u, num_v:', graph.shape)
    if uv==True:
        for u in range(num_u):
            L1paths = u_L1paths[u]
            num_samples = round(L1paths.shape[0] * sample_rate)
            L1_samples_index = np.sort(np.random.choice(L1paths.shape[0], num_samples, replace=False)) # 选择的索引列表
            # L1paths_samples = L1paths[L1_samples_index]
            # L1neighbors = L1paths_samples[:, 0]
            L2paths = np.empty([num_samples,3], dtype=int)

            for i in range(num_samples):
                r1,n1 = L1paths[L1_samples_index[i]]
                L2_rn = v_L1paths[n1]
                L2_sample_index = np.random.choice(L2_rn.shape[0], 1, replace=False)[0]
                r2,n2 = L2_rn[L2_sample_index]
                L2paths[i] = [r1, r2, n2]
            nodes_L2paths.append(L2paths)
    else:
        for v in range(num_v):
            L1paths = v_L1paths[v]
            num_samples = round(L1paths.shape[0] * sample_rate)
            L1_samples_index = np.sort(np.random.choice(L1paths.shape[0], num_samples, replace=False)) # 选择的索引列表
            L2paths = np.empty([num_samples,3], dtype=int)
            for i in range(num_samples):
                r1,n1 = L1paths[L1_samples_index[i]]
                L2_rn = u_L1paths[n1]
                L2_sample_index = np.random.choice(L2_rn.shape[0], 1, replace=False)[0]
                r2,n2 = L2_rn[L2_sample_index]
                L2paths[i] = [r1, r2, n2]
            nodes_L2paths.append(L2paths)

    return nodes_L2paths

def getUAttrList(path_attr, num_u):
    ua_list = []
    for i in range(num_u):
        ua_list.append([0])
    return ua_list

def getVAttrList(path_attr, num_v):
    va_list = []
    ua = open(path_attr, 'r', encoding='utf-8')
    for line in ua:
        nodeId, attr = line.strip().split(';')
        # convert attr into a list of int.
        # if this node has no attr, let attr = [0]
        if attr == '':
            attr = [0]
        else:
            attr = list(map(int, attr.split(',')))
        va_list.append(attr)
    return va_list


def get_train_test(path_edges, test_size):
    """
    generate edge list, label list for train and test
    :param path_edges: path of graph-edges file
    :param test_size: proportion of testset in wholeset
    :return: train edge list, train label list,
             test edge list, test label list in numpy.ndarray format
    """
    ratings = open(path_edges, 'r', encoding='utf-8')
    uv_list = []
    label_list = []
    for line in ratings:
        u, v, label = line.strip().split(';')
        uv_list.append([int(u), int(v)])
        label_list.append(int(label))

    uv_list = np.array(uv_list, dtype=int)
    label_list = np.array(label_list, dtype=int)
    train_uv_list, test_uv_list, train_label_list, test_label_list = train_test_split(uv_list, label_list,test_size=test_size, random_state=1)
    print('trainset and testset splited.')
    print('train samples: {0}; test samples: {1}\n'.format(len(train_label_list), len(test_label_list)))
    return train_uv_list, test_uv_list, train_label_list, test_label_list

if __name__=='__main__':
    path_wholeset_ratings = './data/wholeset_ratings.txt'
    path_wholeset_movies = './data/wholeset_movies.txt'
    train_uv, test_uv, train_rating, test_rating = get_train_test(path_wholeset_ratings, 0.2)
    trainset = torch.utils.data.TensorDataset(torch.tensor(train_uv[:, 0]), torch.tensor(train_uv[:, 1]),
                                              torch.tensor(train_rating))
    testset = torch.utils.data.TensorDataset(torch.tensor(test_uv[:, 0]), torch.tensor(test_uv[:, 1]),
                                             torch.tensor(test_rating))
    Graph = getGraphFromFile(path_wholeset_ratings, 610, 9724)
    uL1paths = getL1paths(Graph, uv=True)
    vL1paths = getL1paths(Graph, uv=False)
    uL2paths = getL2paths(Graph, uL1paths, vL1paths, 0.8, uv=True)
    vL2paths = getL2paths(Graph, uL1paths, vL1paths, 0.8, uv=False)
    print([len(paths) for paths in uL2paths])
    print([len(paths) for paths in vL2paths])
    # Advice: add distribution analysis
    # print([len(paths) for paths in uL1paths])
    # print([len(paths) for paths in vL1paths])
    # print(uL1paths[0])
