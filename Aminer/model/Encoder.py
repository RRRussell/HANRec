# -*- coding: utf-8 -*-
# @Time    : 2020/4/11 18:05
# @Author  : Aurora
# @File    : Encoder.py
# @Function: 

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np

def get_ap_feature(path):
    file = open(path, 'r', encoding='utf-8')
    f = []
    for line in file:
        f.append(torch.tensor(np.array(list(line.split()),dtype=float), dtype=torch.float))
    return f

class Encoder(nn.Module):

    def __init__(self, cuda="cpu"):

        super(Encoder, self).__init__()

        self.embed_dim = 300
        self.device = cuda
        self.linear1 = nn.Linear(3 * self.embed_dim, self.embed_dim)
        self.alpha = nn.Linear(2*self.embed_dim, 1)
        self.beta = nn.Linear(2*self.embed_dim, 1)
        self.gamma = nn.Linear(self.embed_dim, self.embed_dim)
        self.num_author = 16604
        self.num_paper = 12455

        G = nx.read_gml('./link_pred_data/origin_data/clean_G.gml')
        Neighbors = []
        Neighbor = []
        for node in G.nodes():
            for nei in G.neighbors(node):
                Neighbor.append(int(nei))
            Neighbor.sort()
            Neighbors.append(Neighbor)
            Neighbor = []
        self.Neighbors = Neighbors

        a_f = get_ap_feature("./link_pred_data/origin_data/author_vectors_300.txt")
        p_f = get_ap_feature("./link_pred_data/origin_data/paper_vectors_300.txt")
        all_embeddings = a_f+p_f
        self.all_embeddings = all_embeddings

    def forward(self, nodes):
        num_nodes = len(nodes)
        all_combined = []
        for node in nodes:
            neighbors = self.Neighbors[node]
            type_1 = []
            type_2 = []
            if node < self.num_author:
                for neighbor in neighbors:
                    if neighbor < self.num_author:
                        type_1.append(self.all_embeddings[neighbor])
                    else:
                        type_2.append(self.all_embeddings[neighbor])
            else:
                for neighbor in neighbors:
                    if neighbor >= self.num_author:
                        type_1.append(self.all_embeddings[neighbor])
                    else:
                        type_2.append(self.all_embeddings[neighbor])
            # print("type_1",len(type_1))
            # print("type_2", len(type_2))
            if len(type_1) != 0:
                type_1_feat = torch.cat(type_1).reshape(-1, self.embed_dim).float()
                type_1_feat = torch.mean(type_1_feat, 0).reshape(1, self.embed_dim)
            else:
                type_1_feat = torch.ones(1,self.embed_dim)
            if len(type_2) != 0:
                type_2_feat = torch.cat(type_2).reshape(-1, self.embed_dim).float()
                type_2_feat = torch.mean(type_2_feat, 0).reshape(1, self.embed_dim)
            else:
                type_2_feat = torch.ones(1,self.embed_dim)
            self_feats = self.all_embeddings[node].reshape(1, self.embed_dim)

            # alphaa = F.prelu(self.alpha(torch.cat([self_feats, type_1_feat], dim=1)), torch.tensor(1).float())
            # betaa = F.prelu(self.beta(torch.cat([self_feats, type_2_feat], dim=1)), torch.tensor(1).float())
            # l1 = alphaa * type_1_feat
            # l2 = betaa * type_2_feat
            # print("typ1",type_1_feat.shape)
            # print("typ2", type_2_feat.shape)
            # print("self",self_feats.shape)

            combined = torch.cat((self_feats,type_1_feat,type_2_feat)).reshape(1,-1).cuda()
            combined = self.linear1(combined)
            # combined = self_feats + l1 + l2
            # combined = F.prelu(self.gamma(combined), torch.tensor(1).float())
            all_combined.append(combined)

        all_combined = torch.cat(all_combined)
        # print("l1:",l1)
        # print("l2:",l2)
        # combined = torch.cat([self_feats, l1neighs_feats, l2neighs_feats], dim=1) # why dim=1? neigh_fests and self_feats are in shape(128,64),
                                                                                  # containing features of a batch of 128 nodes.
        # print("combined before:",combined)
        # combined = F.prelu(self.linear1(combined),torch.tensor(1).float())

        # print("combined after:",combined)

        return all_combined
