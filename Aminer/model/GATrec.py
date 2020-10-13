# -*- coding: utf-8 -*-
# @Time    : 2020/4/11 18:19
# @Author  : Aurora
# @File    : GATrec.py
# @Function: 

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.utils import get_ap_feature, load_embedding

class GATrec(nn.Module):

    def __init__(self, enc, embed_dim):
        super(GATrec, self).__init__()
        self.enc = enc
        self.embed_dim = embed_dim

        # self.w_1 = nn.Linear(self.embed_dim, 128)
        # self.w_2 = nn.Linear(128, 32)
        # self.w_3 = nn.Linear(32, 8)

        # self.bn1 = nn.BatchNorm1d(128, momentum=0.5)
        # self.bn2= nn.BatchNorm1d(32, momentum=0.5)

        proj_layers = []
        # D = embed_dim
        D = 256
        reduce_factor = 4
        while D > 2:
            next_D = D // reduce_factor
            if next_D < 2:
                next_D = 2
            linear_layer = nn.Linear(D, next_D, bias=True)
            nn.init.xavier_normal_(linear_layer.weight)
            proj_layers.append(linear_layer)
            # proj_layers.append(nn.BatchNorm1d(next_D, momentum=0.5))
            if next_D != 2:
                proj_layers.append(nn.PReLU(1))
            D = next_D

        proj_layers.append(nn.Softmax(dim=1))
        
        self.proj_layers = nn.ModuleList(proj_layers)

        # self.criterion = nn.MSELoss()
        # self.criterion = Focal_Loss(num_class=8)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, nodes_u, nodes_v):

        embeds_u = self.enc(nodes_u)
        embeds_v = self.enc(nodes_v)

        # a_f = get_ap_feature("./link_pred_data/origin_data/author_vectors_300.txt")
        # p_f = get_ap_feature("./link_pred_data/origin_data/paper_vectors_300.txt")
        # all_embeddings = a_f+p_f

        # all_embeddings = np.load("./link_pred_data/baselines/node2vec.npy",allow_pickle=True).item()
        # embeds_u = load_embedding(all_embeddings, list(nodes_u))
        # embeds_v = load_embedding(all_embeddings, list(nodes_v))

        embeds = torch.cat((embeds_u,embeds_v),1)
        # print("embeds",embeds)
        scores = embeds
        for proj_layer in self.proj_layers:
            scores = proj_layer(scores)

        # x = F.prelu(self.bn1(self.w_1(embeds)),torch.tensor(1).float())
        # x = F.dropout(x, training=self.training)
        # x = F.prelu(self.bn2(self.w_2(x)),torch.tensor(1).float())
        # x = F.dropout(x, training=self.training)
        # x = (self.w_3(x))
        # x = F.dropout(x, training=self.training)
        # scores = torch.sigmoid((x))

        # scores = torch.softmax(scores,1)
        # _, scores = torch.max(scores)
        # s = scores.argmax(dim=1)
        return scores#.squeeze()

    def loss(self, nodes_u, nodes_v, labels_list):
        scores = self.forward(nodes_u, nodes_v)
        # print("scores:",(scores))
        # labels_list.long()
        # print("label:",labels_list.long())
        return self.criterion((scores), labels_list.long().reshape(-1))