# -*- coding: utf-8 -*-
# @Time    : 2020/4/11 18:05
# @Author  : Aurora
# @File    : Encoder.py
# @Function: 

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):

    def __init__(self, node2e, embed_dim, l1paths_list, l2paths_list, ua_list, va_list, l1aggr, l2aggr, cuda="cpu", uv=True):
        """
        :param node2e:
        :param embed_dim:
        :param l1paths_list:
        [
            array([ # node
                [r1, n1], # level-1 path
                [r1, n1],
                [r1, n1],
            ]),
        ]
        :param l2paths_list:
        [
            array([ # node
                [r1, r2, n2], # level-2 path
                [r1, r2, n2],
                [r1, r2, n2],
            ]),
        ]
        :param ua_list:
        :param va_list:
        :param l1aggr:
        :param l2aggr:
        :param cuda:
        :param uv:
        """
        super(Encoder, self).__init__()
        self.node2e = node2e
        self.l1paths_list = l1paths_list
        self.l2paths_list = l2paths_list
        self.ua_list = ua_list
        self.va_list = va_list
        self.l1aggr = l1aggr
        self.l2aggr =l2aggr
        self.embed_dim = embed_dim
        self.device = cuda
        self.uv = uv
        self.linear1 = nn.Linear(3 * self.embed_dim, self.embed_dim)
        self.alpha = nn.Linear(2*self.embed_dim, 1)
        self.beta = nn.Linear(2*self.embed_dim, 1)
        self.gamma = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, nodes):
        num_nodes = len(nodes)
        nodes_l1paths = []
        nodes_l2paths = []
        """
        nodes_l1n_attrs: 
        [
            [ # node
                [attr1,attr2,attr3], # attributes of node's neighbor1
                [attr1,attr5],
            ]
        ]
        """
        nodes_l1n_attrs = []
        nodes_l2n_attrs = []

        for i, node in enumerate(nodes):
            nodes_l1paths.append(self.l1paths_list[node])
            nodes_l2paths.append(self.l2paths_list[node])
            if self.uv == True:
                nodes_l1n_attrs.append([self.va_list[neighbor] for neighbor in nodes_l1paths[i][:,1]])
                nodes_l2n_attrs.append([self.ua_list[neighbor] for neighbor in nodes_l2paths[i][:,2]])
            else:
                nodes_l1n_attrs.append([self.ua_list[neighbor] for neighbor in nodes_l1paths[i][:,1]])
                nodes_l2n_attrs.append([self.va_list[neighbor] for neighbor in nodes_l2paths[i][:,2]])

        l1neighs_feats = self.l1aggr.forward(nodes, nodes_l1paths, nodes_l1n_attrs)
        l2neighs_feats = self.l2aggr.forward(nodes, nodes_l2paths, nodes_l2n_attrs)
        self_feats = self.node2e.weight[nodes]
        # self-connection could be considered.
        # print("self_feats",self_feats)
        # print("l1neighs_feats",l1neighs_feats)
        # print("l2neighs_feats",l2neighs_feats)
        # alphaa = torch.sum(self_feats*l1neighs_feats).reshape(-1,1)
        # betaa = torch.sum(self_feats*l2neighs_feats).reshape(-1,1)
        alphaa = F.prelu(self.alpha(torch.cat([self_feats, l1neighs_feats], dim=1)),torch.tensor(1).float())
        betaa = F.prelu(self.beta(torch.cat([self_feats, l2neighs_feats], dim=1)),torch.tensor(1).float())
        l1 = alphaa*l1neighs_feats
        l2 = betaa*l2neighs_feats
        combined = self_feats+l1+l2

        # print("l1:",l1)
        # print("l2:",l2)
        # combined = torch.cat([self_feats, l1neighs_feats, l2neighs_feats], dim=1) # why dim=1? neigh_fests and self_feats are in shape(128,64),
                                                                                  # containing features of a batch of 128 nodes.
        # print("combined before:",combined)
        # combined = F.prelu(self.linear1(combined),torch.tensor(1).float())
        combined = F.prelu(self.gamma(combined),torch.tensor(1).float())
        # print("combined after:",combined)

        return combined
