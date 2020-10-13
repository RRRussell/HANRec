# -*- coding: utf-8 -*-
# @Time    : 2020/4/15 19:40
# @Author  : Aurora
# @File    : data_loader_new.py
# @Function: 

import numpy as np
import torch
import torch.utils.data
from sklearn.model_selection import train_test_split
# from model.L1neighs_Aggregator import L1neighs_Aggregator

num_author = 16604
num_paper = 12455

def get_ap_feature(path):
    file = open(path, 'r', encoding='utf-8')
    f = []
    for line in file:
        f.append(torch.tensor(np.array(list(line.split()), dtype=float)))
    return f

def getGraphFromFile(path_edges, num_u, num_v):
    ratings = open(path_edges, 'r', encoding='utf-8')
    G = np.ones([num_u, num_v], dtype=int)
    G = -G
    for line in ratings:
        u, v = line.strip().split(' ')
        G[int(u), int(v)] = 1
    return G

def getGraphFromArray(edges, num_u, num_v):
    G = np.ones([num_u, num_v], dtype=int)
    G = -G
    for i, edge in enumerate(edges):
        u, v = edge
        G[int(u), v] = label
    return G

def get_edge_and_label(path, no=False):
    edge = []
    label = []
    with open(path) as f:
        for line in f:
            u, v = line.split(" ")
            u = int(u)
            v = int(v.replace("\n", ""))
            edge.append([u,v])
            if no:
                label.append([0])
            else:
                label.append([1])
    return edge, label

def get_train_test_withcount():

    train_edge_list, train_label_list = get_edge_and_label("./link_pred_data/link_pred/train_edge.txt")
    train_no_edge_list, train_no_label_list = get_edge_and_label("./link_pred_data/link_pred/train_no_edge.txt",no=True)

    train_edge_list=train_edge_list+train_no_edge_list
    train_label_list=train_label_list+train_no_label_list

    test_edge_list, test_label_list = get_edge_and_label("./link_pred_data/link_pred/test_edge.txt")
    test_no_edge_list, test_no_label_list = get_edge_and_label("./link_pred_data/link_pred/test_no_edge.txt",no=True)

    test_edge_list=test_edge_list+test_no_edge_list
    test_label_list=test_label_list+test_no_label_list

    import random
    randnum = random.randint(0,100)
    random.seed(randnum)
    random.shuffle(train_edge_list)
    random.seed(randnum)
    random.shuffle(train_label_list)

    randnum = random.randint(0,100)
    random.seed(randnum)
    random.shuffle(test_edge_list)
    random.seed(randnum)
    random.shuffle(test_label_list)
    
    # count = 0
    count = 1280
    if count != 0:
        train_edge_list = train_edge_list[:count]
        train_label_list = train_label_list[:count]
        test_edge_list = test_edge_list[:count]
        test_label_list = test_label_list[:count]

    train_edge_list = np.array(train_edge_list)
    train_label_list = np.array(train_label_list)
    test_edge_list = np.array(test_edge_list)
    test_label_list = np.array(test_label_list)

    print('trainset and testset splited.')
    print('train samples: {0}; test samples: {1}\n'.format(len(train_label_list), len(test_label_list)))
    return train_edge_list, test_edge_list, train_label_list, test_label_list

def getL1paths(graph, ap):
    num_u, num_v = graph.shape
    nodes_L1paths = []
    print('num_u, num_v:', graph.shape)
    if ap == "ap":
        for u in range(num_u):
            connection = graph[u, :]
            L1neighbors = np.where(connection!=-1)[0] + num_author# array([vid, vid, vid, ...])
            num_neighbors = len(L1neighbors)
            if num_neighbors == 0:
                nodes_L1paths.append(np.array([u]))
                continue
            nodes_L1paths.append(L1neighbors)
    elif ap == "pa":
        for v in range(num_v):
            connection = graph[:, v]
            L1neighbors = np.where(connection!=-1)[0] # array([uid, uid, uid, ...])
            num_neighbors = len(L1neighbors)
            if num_neighbors == 0:
                nodes_L1paths.append(np.array([v]))
                continue
            nodes_L1paths.append(L1neighbors)
    elif ap == "aa":
        for u in range(num_u):
            connection = graph[u, :]
            L1neighbors = np.where(connection!=-1)[0] # array([vid, vid, vid, ...])
            num_neighbors = len(L1neighbors)
            if num_neighbors == 0:
                nodes_L1paths.append(np.array([u]))
                continue
            nodes_L1paths.append(L1neighbors)
    elif ap == "pp":
        for u in range(num_u):
            connection = graph[u, :]
            L1neighbors = np.where(connection!=-1)[0] + num_author# array([vid, vid, vid, ...])
            num_neighbors = len(L1neighbors)
            if num_neighbors == 0:
                nodes_L1paths.append(np.array([u+num_author]))
                continue
            nodes_L1paths.append(L1neighbors)
    
    return nodes_L1paths

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

    a_f = get_ap_feature("./link_pred_data/origin_data/author_vectors.txt")
    p_f = get_ap_feature("./link_pred_data/origin_data/paper_vectors.txt")

    path_class_a = "./link_pred_data/origin_data/author_class.txt"
    path_class_p = "./link_pred_data/origin_data/paper_class.txt"
    test_size = 0.2

    train_ap, test_ap, train_class, test_class = get_train_test_withcount(path_class_a, path_class_p, test_size)

    G_ap = getGraphFromFile("./link_pred_data/origin_data/Aminer_author2paper.txt", num_author, num_paper)
    G_aa = getGraphFromFile("./link_pred_data/origin_data/Aminer_coauthor.txt", num_author, num_author)
    G_pp = getGraphFromFile("./link_pred_data/origin_data/Aminer_citation.txt", num_paper, num_paper)

    # print(G_ap)
    # print(G_aa)
    # print(G_pp)

    ap_L1path = getL1paths(G_ap, ap="ap")
    # print(ap_L1path)
    pa_L1path = getL1paths(G_ap, ap="pa")
    # print(pa_L1path)
    aa_L1path = getL1paths(G_aa, ap="aa")
    # print(aa_L1path)
    pp_L1path = getL1paths(G_pp, ap="pp")
    print(pp_L1path)

    # Advice: add distribution analysis
    # print([len(paths) for paths in uL1paths])
    # print([len(paths) for paths in vL1paths])
    # print(uL1paths[0])
