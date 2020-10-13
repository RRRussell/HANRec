# -*- coding: utf-8 -*-
# @Time    : 2020/4/11 18:05
# @Author  : Aurora
# @File    : L2neighs_Aggregator.py
# @Function: 


import torch
import torch.nn as nn
import torch.nn.functional as F
from model.Attention import Attention


class L2neighs_Aggregator(nn.Module):
    """
    to aggregate embeddings of level-2 neighbors (indirect neighbors).
    """

    def __init__(self, u2e, v2e, r2e, ua2e, va2e, embed_dim, cuda="cpu", uv=True):
        super(L2neighs_Aggregator, self).__init__()
        self.u2e = u2e
        self.v2e = v2e
        self.r2e = r2e
        self.ua2e = ua2e
        self.va2e = va2e
        self.embed_dim = embed_dim
        self.device = cuda
        self.uv = uv

        self.w_r1 = nn.Linear(self.embed_dim * 4, self.embed_dim * 2)
        self.w_r2 = nn.Linear(self.embed_dim * 2, self.embed_dim)
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

    def forward(self, nodes, nodes_l2paths, nodes_l2n_attrs):
        """
        :param nodes:
        :param nodes_l2paths:
        [
            array([ # node
                [r1, r2, n2], # level-2 path
                [r1, r2, n2],
                [r1, r2, n2],
            ]),
        ]
        :param nodes_l2n_attrs:
        :return:
        """
        embed_matrix = torch.empty(len(nodes), self.embed_dim, dtype=torch.float).to(self.device)
        for i in range(len(nodes)):
            paths = nodes_l2paths[i]
            if len(paths) == 0:
                embed_matrix[i] = torch.zeros(self.embed_dim, dtype=torch.float).to(self.device)
            else:
                l2neighbors = paths[:,2]
                l2neighbors_r1 = paths[:,0]
                l2neighbors_r2 = paths[:,1]
                r1_es = self.r2e.weight[l2neighbors_r1]
                r2_es = self.r2e.weight[l2neighbors_r2]

                if self.uv == True:
                    l2neighs_es = self.u2e.weight[l2neighbors]
                    l2neighs_attr_es = self.uvAttrEmbedding(nodes_l2n_attrs[i], uv=True)
                    self_e = self.u2e.weight[nodes[i]]
                else:
                    l2neighs_es = self.v2e.weight[l2neighbors]
                    l2neighs_attr_es = self.uvAttrEmbedding(nodes_l2n_attrs[i], uv=False)
                    self_e = self.v2e.weight[nodes[i]]

                x = torch.cat((r1_es, r2_es, l2neighs_es,l2neighs_attr_es), 1)
                x = F.relu(self.w_r1(x))
                o_l2neighs_es = F.relu(self.w_r2(x))

                att_w = self.att(o_l2neighs_es, self_e, len(paths))
                att_l2neighs_e = torch.mm(o_l2neighs_es.t(), att_w)
                att_l2neighs_e = att_l2neighs_e.t()
                embed_matrix[i] = att_l2neighs_e

        return embed_matrix