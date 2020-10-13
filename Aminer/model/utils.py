import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def get_ap_feature(path):
    file = open(path, 'r', encoding='utf-8')
    f = []
    for line in file:
        f.append(torch.tensor(np.array(list(line.split()), dtype=float)))
    return f

def load_embedding(all_embeddings, nodes):
    temp = all_embeddings[str(int(nodes[0]))]
    for i in range(1,len(nodes)):
        temp = np.vstack((temp,all_embeddings[str(int(nodes[i]))]))
    return torch.tensor(temp)