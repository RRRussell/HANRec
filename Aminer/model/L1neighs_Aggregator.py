# -*- coding: utf-8 -*-
# @Time    : 2020/4/11 18:04
# @Author  : Aurora
# @File    : L1neighs_Aggregator.py
# @Function: 

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.Attention import Attention

class L1neighs_Aggregator(nn.Module):
    """
    to aggregate embeddings of level-1 neighbors (direct neighbors).
    """

    def __init__(self, a2e, p2e, ap, embed_dim, cuda="cpu"):
        super(L1neighs_Aggregator, self).__init__()
        self.a2e = a2e
        self.p2e = p2e
        self.ap = ap
        self.embed_dim = embed_dim
        self.device = cuda

        self.att = Attention(self.embed_dim)

        self.num_author = 16604
        self.num_paper = 12455
        
    def forward(self, node, node_l1path):

        neighbors = node_l1path[node]
        num_neighbors = len(neighbors)
        # print(neighbors)
        # print(type(neighbors))
        if num_neighbors == 0:
            l1_feats = torch.zeros(self.embed_dim, dtype=torch.float).to(self.device)
        else:
            neighs_es = []
            if self.ap == "ap":
                for n in neighbors:
                    neighs_es.append(self.p2e[n-self.num_author])
                # neighs_es = self.p2e[neighbors-self.num_author]
                self_e = self.a2e[node]

            elif self.ap == "pa":
                for n in neighbors:
                    neighs_es.append(self.a2e[n])
                # neighs_es = self.a2e[neighbors]
                self_e = self.p2e[node]

            elif self.ap == "aa":
                for n in neighbors:
                    neighs_es.append(self.a2e[n])
                # neighs_es = self.a2e[neighbors]
                self_e = self.a2e[node]

            elif self.ap == "pp":
                for n in neighbors:
                    neighs_es.append(self.p2e[n-self.num_author])
                # neighs_es = self.p2e[neighbors-self.num_author]
                self_e = self.p2e[node]

            neighs_es = torch.cat(neighs_es).reshape(num_neighbors,-1).float()
            # neighs_es = torch.tensor(neighs_es, dtype=torch.float)

            # att_w = self.att(neighs_es, self_e, num_neighbors)
            # att_neighs_e = torch.mm(neighs_es.t(), att_w)
            # l1_feats = att_neighs_e.t()
            l1_feats = torch.mean(neighs_es,0)

        return l1_feats

        # if self.ap == "ap":
        #     neighbors = node_l1path[node]
        #     if len(neighbors) == 0:
        #         l1_feats = torch.zeros(self.embed_dim, dtype=torch.float).to(self.device)
        #     else:
        #         neighs_es = self.p2e[neighbors-self.num_author]
        #         self_e = self.a2e[node]

        # elif self.ap == "pa":
        #     neighbors = node_l1path[node]
        #     if len(neighbors) == 0:
        #         l1_feats = torch.zeros(self.embed_dim, dtype=torch.float).to(self.device)
        #     else:
        #         neighs_es = self.a2e[neighbors]
        #         self_e = self.p2e[node]

        # elif self.ap == "aa":
        #     neighbors = node_l1path[node]
        #     if len(neighbors) == 0:
        #         l1_feats = torch.zeros(self.embed_dim, dtype=torch.float).to(self.device)
        #     else:
        #         neighs_es = self.a2e[neighbors]
        #         self_e = self.a2e[node]

        # elif self.ap == "pp":
        #     print("neighbors",neighbors)
        #     neighbors = node_l1path[node]
        #     if len(neighbors) == 0:
        #         l1_feats = torch.zeros(self.embed_dim, dtype=torch.float).to(self.device)
        #     else:
        #         neighs_es = self.p2e[neighbors-self.num_author]
        #         self_e = self.p2e[node-self.num_author]

        # if len(neighbors) != 0:
        #     att_w = self.att(neighs_es, self_e, num_neighbors)
        #     att_neighs_e = torch.mm(neighs_es.t(), att_w)
        #     l1_feats = att_neighs_e.t()

        # return l1_feats