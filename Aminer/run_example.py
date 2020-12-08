# -*- coding: utf-8 -*-
# @Time    : 2020/4/8 09:12
# @Author  : Aurora
# @File    : run_example.py
# @Function: example of GNN for recommendation

import argparse
import os
import sys
import numpy as np
import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.utils.data
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt

from sklearn.metrics import roc_curve, auc, accuracy_score

# import data_loader
import data_loader_new
from model.L1neighs_Aggregator import L1neighs_Aggregator
from model.L2neighs_Aggregator import L2neighs_Aggregator
from model.Encoder import Encoder
from model.GATrec import GATrec

def train(model, train_loader, optimizer, epoch, best_auc, best_acc, device):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        batch_start = time.time()
        batch_nodes_u, batch_nodes_v, labels_list = data
        optimizer.zero_grad()
        loss = model.loss(batch_nodes_u.to(device), batch_nodes_v.to(device), labels_list.to(device))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 0:
            print('[%d, %5d] loss: %.3f, The best auc/acc: %.6f / %.6f' % (
                epoch, i, running_loss / 100, best_auc, best_acc))
            running_loss = 0.0
        batch_end = time.time()
        # print("loss:",loss)

        # print('batch train time cost:{} s'.format(batch_end-batch_start))
    return 0

def test(model, device, test_loader):
    model.eval()
    auc_pred = []
    acc_pred = []
    target = []
    with torch.no_grad():
        for test_u, test_v, tmp_target in test_loader:
            test_u, test_v, tmp_target = test_u.to(device), test_v.to(device), tmp_target.to(device)
            val_output = model.forward(test_u, test_v)
            auc_val_output = val_output[:, 1]
            acc_val_output = torch.max(val_output,1)[1]
            auc_pred.append(list(auc_val_output.data.cpu().numpy()))
            acc_pred.append(list(acc_val_output.data.cpu().numpy()))
            target.append(list(tmp_target.data.cpu().numpy()))

    auc_pred = np.array(sum(auc_pred, [])).reshape(-1)
    acc_pred = np.array(sum(acc_pred, [])).reshape(-1)
    target = np.array(sum(target, [])).reshape(-1)

    print("predict", auc_pred.reshape(-1)[:20])
    print("target", target.reshape(-1)[:20])

    fpr, tpr, thresholds = roc_curve(target, auc_pred, pos_label=1)
    auc_value = auc(fpr, tpr)

    acc_value = accuracy_score(target, acc_pred)
    # print("-----sklearn:",auc(fpr, tpr))
    # expected_rmse = sqrt(mean_squared_error(tmp_pred, target))
    # mae = mean_absolute_error(tmp_pred, target)
    return auc_value, acc_value

def main():
    parser = argparse.ArgumentParser(description='weihaoGNN')
    parser.add_argument('-bs', '--batch_size', type=int, default=128, help='input batch size for training')
    parser.add_argument('-ed', '--embed_dim', type=int, default=128, help='embedding size')
    parser.add_argument('-lr', '--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('-tbs', '--test_batch_size', type=int, default=1000, help='input batch size for testing')
    parser.add_argument('-ep', '--epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('-r', '--l2rate', type=float, default=1.0, help='rate of selected l2 paths')
    args = parser.parse_args()

    now = datetime.now()
    now = now.strftime('%m%d_%H%M%S')

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    batch_size = args.batch_size
    test_batch_size = args.test_batch_size
    embed_dim = args.embed_dim
    lr = args.lr
    epochs = args.epochs
    # c = args.count
    r = args.l2rate

    print('#### Ziheng Aminer ####\n')
    print('parameter:\n{}'.format(args.__dict__))

    # train_uv, test_uv, train_rating, test_rating = data_loader_new.get_train_test(path_ratings, 0.2)
    train_uv, test_uv, train_label, test_label = data_loader_new.get_train_test_withcount()

    # test_uv = train_uv
    # test_rating = train_rating

    # G = data_loader_new.getGraphFromArray(train_uv, train_label, 610, 9724)
    # uL1paths = data_loader_new.getL1paths(G, uv=True)
    # vL1paths = data_loader_new.getL1paths(G, uv=False)
    # uL2paths = data_loader_new.getL2paths(G, uL1paths, vL1paths, r, uv=True)
    # vL2paths = data_loader_new.getL2paths(G, uL1paths, vL1paths, r, uv=False)
    # ua_list = data_loader_new.getUAttrList('', 610)
    # va_list = data_loader_new.getVAttrList(path_movies, 9724)

    trainset = torch.utils.data.TensorDataset(torch.LongTensor(train_uv[:, 0]), torch.LongTensor(train_uv[:, 1]), torch.FloatTensor(train_label))
    testset = torch.utils.data.TensorDataset(torch.LongTensor(test_uv[:, 0]), torch.LongTensor(test_uv[:, 1]), torch.FloatTensor(test_label))
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=True)

    # initialize embedding for subsequent training
    # u2e = nn.Embedding(num_users, embed_dim).to(device)
    # v2e = nn.Embedding(num_movies, embed_dim).to(device)
    # r2e = nn.Embedding(num_ratings, embed_dim).to(device)

    # initialzie attribute embedding to distinguish different attributes
    # va2e, ua2e would never change in training process
    # ua2e = nn.Embedding(num_user_attr, embed_dim).to(device)
    # va2e = nn.Embedding(num_genres, embed_dim).to(device)

    # u2e.weight.requires_grad=False
    # v2e.weight.requires_grad=False
    # r2e.weight.requires_grad=False

    # ua2e.weight.requires_grad=False
    # va2e.weight.requires_grad=False

    # u_L1Aggregator = L1neighs_Aggregator(u2e, v2e, r2e, ua2e, va2e, embed_dim, cuda=device, uv=True)
    # v_L1Aggregator = L1neighs_Aggregator(u2e, v2e, r2e, ua2e, va2e, embed_dim, cuda=device, uv=False)
    # u_L2Aggregator = L2neighs_Aggregator(u2e, v2e, r2e, ua2e, va2e, embed_dim, cuda=device, uv=True)
    # v_L2Aggregator = L2neighs_Aggregator(u2e, v2e, r2e, ua2e, va2e, embed_dim, cuda=device, uv=False)

    # u_Encoder = Encoder(u2e, embed_dim, uL1paths, uL2paths, ua_list, va_list, u_L1Aggregator, u_L2Aggregator, cuda=device, uv=True)
    # v_Encoder = Encoder(v2e, embed_dim, vL1paths, vL2paths, ua_list, va_list, v_L1Aggregator, v_L2Aggregator, cuda=device, uv=False)
    Rec_Encoder = Encoder()

    gatRec = GATrec(Rec_Encoder, embed_dim).to(device)
    # optimizer = torch.optim.RMSprop(gatRec.parameters(), lr=lr, alpha=0.99)
    optimizer = torch.optim.Adam(gatRec.parameters(), lr=lr)

    best_auc = 0
    best_acc = 0
    endure_count = 0

    for epoch in range(1, epochs + 1):
        train(gatRec, train_loader, optimizer, epoch, best_auc, best_acc, device)
        auc, acc = test(gatRec, device, test_loader)
        # please add the validation set to tune the hyper-parameters based on your datasets.
        # early stopping (no validation set in toy dataset)
        if best_auc <= auc:
            best_auc = auc
            best_acc = acc
            endure_count = 0
        else:
            endure_count += 1
        print("auc: %.4f" % auc)

        if endure_count > 5:
            break

    print('Finished.')


if __name__ == '__main__':
    main()

