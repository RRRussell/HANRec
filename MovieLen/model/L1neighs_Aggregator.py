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

    def __init__(self, u2e, v2e, r2e, ua2e, va2e, embed_dim, cuda="cpu", uv=True):
        super(L1neighs_Aggregator, self).__init__()
        self.u2e = u2e
        self.v2e = v2e
        self.r2e = r2e
        self.ua2e = ua2e
        self.va2e = va2e
        self.embed_dim = embed_dim
        self.device = cuda
        self.uv = uv

        self.w_r1 = nn.Linear(self.embed_dim * 3, self.embed_dim)
        self.w_r2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.att = Attention(self.embed_dim)

    def uvAttrEmbedding(self, nodes_attrs, uv):
        num_nodes = len(nodes_attrs)
        embed_nodes_attr = torch.empty(num_nodes, self.embed_dim, dtype=torch.float).to(self.device)
        if uv==True:
            for i in range(num_nodes):
                attrs_es = self.ua2e.weight[nodes_attrs[i]]
                attr_e = torch.sum(attrs_es, 0)
                embed_nodes_attr[i] = attr_e
        else:
            for i in range(num_nodes):
                attrs_es = self.va2e.weight[nodes_attrs[i]]
                attr_e = torch.sum(attrs_es, 0)
                embed_nodes_attr[i] = attr_e
        return embed_nodes_attr

    def forward(self, nodes, nodes_l1paths, nodes_l1n_attrs):
        """
        :param nodes: [node1,node2,node3,...]
        :param nodes_l1paths:
        [
            array([ # node
                [r1, n1], # level-1 path
                [r1, n1],
                [r1, n1],
            ]),
        ]
        :param nodes_l1n_attrs:
        [
            [ # node
                [attr1,attr2,attr3], # attributes of node's neighbor1
                [attr1,attr5],
            ],
        ]
        :return:
        """
        embed_matrix = torch.empty(len(nodes), self.embed_dim, dtype=torch.float).to(self.device)
        for i in range(len(nodes)):
            paths = nodes_l1paths[i]
            if len(paths) == 0:
                embed_matrix[i] = torch.zeros(self.embed_dim, dtype=torch.float).to(self.device)
                continue
            neighbors = paths[:,1]
            neighbors_r = paths[:,0]
            num_neighbors = len(neighbors)

            if self.uv == True:
                # user component
                neighs_es = self.v2e.weight[neighbors]
                neighs_attr_es = self.uvAttrEmbedding(nodes_l1n_attrs[i], uv=False)
                self_e = self.u2e.weight[nodes[i]]
            else:
                # item component
                neighs_es = self.u2e.weight[neighbors]
                neighs_attr_es = self.uvAttrEmbedding(nodes_l1n_attrs[i], uv=True)
                self_e = self.v2e.weight[nodes[i]]

            r_es = self.r2e.weight[neighbors_r]
            x = torch.cat((neighs_es, r_es, neighs_attr_es), 1)
            x = F.relu(self.w_r1(x))
            o_neighs_es = F.relu(self.w_r2(x))

            att_w = self.att(o_neighs_es, self_e, num_neighbors)
            att_neighs_e = torch.mm(o_neighs_es.t(), att_w)
            att_neighs_e = att_neighs_e.t()
            embed_matrix[i] = att_neighs_e

        batch_att_neighs_es = embed_matrix
        return batch_att_neighs_es


"""
:param nodes: a batch of nodes
:param nodes_n: nodes's neighbors
:param nodes_nr: corresponding ratings of nodes' neighbors
:param nodes_na: corresponding attributes of nodes' neighbors
:return:
"""