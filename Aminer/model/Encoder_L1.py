# -*- coding: utf-8 -*-
# @Time    : 2020/4/18 14:08
# @Author  : Aurora
# @File    : Encoder_L1.py
# @Function: 

import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder_L1(nn.Module):

    # def __init__(self, node2e, embed_dim, l1paths_list, ua_list, va_list, l1aggr, cuda="cpu", uv=True):
    def __init__(self, a2e, p2e, embed_dim, ap_L1path, pa_L1path, aa_L1path, pp_L1path, \
                        ap_L1Aggregator, pa_L1Aggregator, aa_L1Aggregator, pp_L1Aggregator, cuda="cpu"):

        super(Encoder_L1, self).__init__()
        self.a2e = a2e
        self.p2e = p2e

        self.embed_dim = embed_dim

        self.ap_L1path = ap_L1path
        self.pa_L1path = pa_L1path
        self.aa_L1path = aa_L1path
        self.pp_L1path = pp_L1path

        self.ap_L1Aggregator = ap_L1Aggregator
        self.pa_L1Aggregator = pa_L1Aggregator
        self.aa_L1Aggregator = aa_L1Aggregator
        self.pp_L1Aggregator = pp_L1Aggregator

        self.device = cuda

        self.linear_self = nn.Linear(300, self.embed_dim)

        self.linear1 = nn.Linear(3 * self.embed_dim, self.embed_dim)

        self.num_author = 16604
        self.num_paper = 13553

    def forward(self, nodes):
        num_nodes = len(nodes)
        self_feats = []
        l1neighs_feats_0 = []
        l1neighs_feats_1 = []

        for i, node in enumerate(nodes):

            if node < self.num_author:
                # node_type = "author"
                self_feats_i = self.a2e[node]
                l1neighs_feats_0i = self.ap_L1Aggregator.forward(node, self.ap_L1path)
                l1neighs_feats_1i = self.aa_L1Aggregator.forward(node, self.aa_L1path)

            else:
                # node_type = "paper"
                self_feats_i = self.p2e[node-self.num_author]
                l1neighs_feats_0i = self.pa_L1Aggregator.forward(node-self.num_author, self.pa_L1path)
                l1neighs_feats_1i = self.pp_L1Aggregator.forward(node-self.num_author, self.pp_L1path)

            self_feats.append(self_feats_i)
            l1neighs_feats_0.append(l1neighs_feats_0i)
            l1neighs_feats_1.append(l1neighs_feats_1i)

        # self-connection could be considered.
        # self_feats = torch.tensor(self_feats, dtype=torch.float)
        # l1neighs_feats_0 = torch.tensor(l1neighs_feats_0, dtype=torch.float)
        # l1neighs_feats_1 = torch.tensor(l1neighs_feats_1, dtype=torch.float)

        self_feats = torch.cat(self_feats).reshape(num_nodes,-1).float()
        # self_feats = F.prelu(self.linear1(self_feats),torch.tensor(1).float())
        # print("l1",len(l1neighs_feats_0))
        # print("l1s",type(l1neighs_feats_0))
        l1neighs_feats_0 = torch.cat(l1neighs_feats_0).reshape(num_nodes,-1).float()
        l1neighs_feats_1 = torch.cat(l1neighs_feats_1).reshape(num_nodes,-1).float()

        combined = torch.cat([self_feats, l1neighs_feats_0, l1neighs_feats_1], dim=1)

        combined = F.prelu(self.linear1(combined),torch.tensor(1).float())

        return combined
        # return self_feats