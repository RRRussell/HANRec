# -*- coding: utf-8 -*-
# @Time    : 2020/4/11 16:02
# @Author  : cite from Wenqi Fan
# @File    : Attention.py
# @Function:

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, embedding_dims):
        super(Attention, self).__init__()
        self.embed_dim = embedding_dims
        # self.bilinear = nn.Bilinear(self.embed_dim, self.embed_dim, 1)
        self.att1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.att2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.att3 = nn.Linear(self.embed_dim, 1)
        self.softmax = nn.Softmax(0)

    def forward(self, neighs_es, node_e, num_neighs):
        node_es = node_e.repeat(num_neighs, 1)
        x = torch.cat((neighs_es, node_es), 1)
        x = F.relu(self.att1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.att2(x))
        x = F.dropout(x, training=self.training)
        x = self.att3(x)
        att = F.softmax(x, dim=0)
        return att

