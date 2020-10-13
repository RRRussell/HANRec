
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gensim.models import Word2Vec
import gensim

def load_embedding(all_embeddings, nodes):
    temp = all_embeddings[str(int(nodes[0]))]
    for i in range(1,len(nodes)):
    	temp = np.vstack((temp,all_embeddings[str(int(nodes[i]))]))
    	
    return torch.tensor(temp)

# all_embeddings = np.load("./DeepWalk.npy").item()
all_embeddings = np.load("./Node2Vec.npy").item()
print(all_embeddings)
u = torch.tensor([3,2,10])
encode_u = load_embedding(all_embeddings, list(u))
print("encode_u",encode_u)

# model = Word2Vec.load("./Node2Vec.emb")
# print(model)
# word2vec = gensim.models.word2vec.Word2Vec.load("./Node2Vec.emb")
# print(word2vec)

# f = open("./Node2Vec.emb","r")
# # print()
# f.readline()
# for line in f:
# 	node, emb = (line.split(' ', 1))
# 	# print(torch.tensor(float(emb)))

