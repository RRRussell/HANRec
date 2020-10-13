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

# import data_loader
import data_loader_new
from logger.Logger import Logger
from model.L1neighs_Aggregator import L1neighs_Aggregator
from model.L2neighs_Aggregator import L2neighs_Aggregator
from model.Encoder import Encoder
from model.GATrec import GATrec


def train(model, train_loader, optimizer, epoch, best_rmse, best_mae, device):
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
            print('[%d, %5d] loss: %.3f, The best rmse/mae: %.6f / %.6f' % (
                epoch, i, running_loss / 100, best_rmse, best_mae))
            running_loss = 0.0
        batch_end = time.time()
        print('batch train time cost:{} s'.format(batch_end-batch_start))
    return 0

def test(model, device, test_loader):
    model.eval()
    tmp_pred = []
    target = []
    with torch.no_grad():
        for test_u, test_v, tmp_target in test_loader:
            test_u, test_v, tmp_target = test_u.to(device), test_v.to(device), tmp_target.to(device)
            val_output = model.forward(test_u, test_v)
            tmp_pred.append(list(val_output.data.cpu().numpy()))
            target.append(list(tmp_target.data.cpu().numpy()))
    tmp_pred = np.array(sum(tmp_pred, []))
    target = np.array(sum(target, []))
    expected_rmse = sqrt(mean_squared_error(tmp_pred, target))
    mae = mean_absolute_error(tmp_pred, target)
    return expected_rmse, mae

def main():
    parser = argparse.ArgumentParser(description='weihaoGNN')
    parser.add_argument('-bs', '--batch_size', type=int, default=128, help='input batch size for training')
    parser.add_argument('-ed', '--embed_dim', type=int, default=64, help='embedding size')
    parser.add_argument('-lr', '--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('-tbs', '--test_batch_size', type=int, default=1000, help='input batch size for testing')
    parser.add_argument('-ep', '--epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('-c', '--count', type=int, default=1000, help='number of edges for training and testing')
    parser.add_argument('-r', '--l2rate', type=float, default=1.0, help='rate of selected l2 paths')
    args = parser.parse_args()

    path_ratings = './data/wholeset_ratings.txt'
    path_movies = './data/wholeset_movies.txt'
    path_log = './log/'
    path_log_err = './log/err/'

    now = datetime.now()
    now = now.strftime('%m%d_%H%M%S')
    sys.stdout = Logger(path_log+'run_example-'+now+'.log', sys.stdout)
    sys.stderr = Logger(path_log_err+'run_example_err-'+now+'.log', sys.stderr)

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
    c = args.count
    r = args.l2rate

    print('#### weihaoGAT ####\n')
    print('parameter:\n{}'.format(args.__dict__))

    # train_uv, test_uv, train_rating, test_rating = data_loader_new.get_train_test(path_ratings, 0.2)
    train_uv, test_uv, train_rating, test_rating = data_loader_new.get_train_test_withcount(path_ratings, 0.2, c)

    G = data_loader_new.getGraphFromArray(train_uv, train_rating, 610, 9724)
    uL1paths = data_loader_new.getL1paths(G, uv=True)
    vL1paths = data_loader_new.getL1paths(G, uv=False)
    uL2paths = data_loader_new.getL2paths(G, uL1paths, vL1paths, r, uv=True)
    vL2paths = data_loader_new.getL2paths(G, uL1paths, vL1paths, r, uv=False)
    ua_list = data_loader_new.getUAttrList('', 610)
    va_list = data_loader_new.getVAttrList(path_movies, 9724)

    ratings_dict = {
        0:0.5,
        1:1.0,
        2:1.5,
        3:2.0,
        4:2.5,
        5:3.0,
        6:3.5,
        7:4.0,
        8:4.5,
        9:5.0,
    }
    movie_genres_dict = {
        'None': '0',
        'Action': '1',
        'Adventure': '2',
        'Animation': '3',
        'Children': '4',
        'Comedy': '5',
        'Crime': '6',
        'Documentary': '7',
        'Drama': '8',
        'Fantasy': '9',
        'Film-Noir': '10',
        'Horror': '11',
        'Musical': '12',
        'Mystery': '13',
        'Romance': '14',
        'Sci-Fi': '15',
        'Thriller': '16',
        'War': '17',
        'Western': '18',
    }
    # in MovieLens-latest-small, users have no attributes
    user_attr_dict = {'None': '0'}

    trainset = torch.utils.data.TensorDataset(torch.LongTensor(train_uv[:, 0]), torch.LongTensor(train_uv[:, 1]), torch.FloatTensor(train_rating))
    testset = torch.utils.data.TensorDataset(torch.LongTensor(test_uv[:, 0]), torch.LongTensor(test_uv[:, 1]), torch.FloatTensor(test_rating))
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=True)

    num_users = 610
    num_movies = 9724
    num_ratings = ratings_dict.__len__()
    num_genres = movie_genres_dict.__len__()
    num_user_attr = user_attr_dict.__len__()

    # initialize embedding for subsequent training
    u2e = nn.Embedding(num_users, embed_dim).to(device)
    v2e = nn.Embedding(num_movies, embed_dim).to(device)
    r2e = nn.Embedding(num_ratings, embed_dim).to(device)

    # initialzie attribute embedding to distinguish different attributes
    # va2e, ua2e would never change in training process
    ua2e = nn.Embedding(num_user_attr, embed_dim).to(device)
    va2e = nn.Embedding(num_genres, embed_dim).to(device)

    u_L1Aggregator = L1neighs_Aggregator(u2e, v2e, r2e, ua2e, va2e, embed_dim, cuda=device, uv=True)
    v_L1Aggregator = L1neighs_Aggregator(u2e, v2e, r2e, ua2e, va2e, embed_dim, cuda=device, uv=False)
    u_L2Aggregator = L2neighs_Aggregator(u2e, v2e, r2e, ua2e, va2e, embed_dim, cuda=device, uv=True)
    v_L2Aggregator = L2neighs_Aggregator(u2e, v2e, r2e, ua2e, va2e, embed_dim, cuda=device, uv=False)

    u_Encoder = Encoder(u2e, embed_dim, uL1paths, uL2paths, ua_list, va_list, u_L1Aggregator, u_L2Aggregator, cuda=device, uv=True)
    v_Encoder = Encoder(v2e, embed_dim, vL1paths, vL2paths, ua_list, va_list, v_L1Aggregator, v_L2Aggregator, cuda=device, uv=False)

    gatRec = GATrec(u_Encoder, v_Encoder, embed_dim).to(device)
    # optimizer = torch.optim.RMSprop(gatRec.parameters(), lr=lr, alpha=0.99)
    optimizer = torch.optim.Adam(gatRec.parameters(), lr=lr)

    best_rmse = 9999.0
    best_mae = 9999.0
    endure_count = 0

    for epoch in range(1, epochs + 1):
        train(gatRec, train_loader, optimizer, epoch, best_rmse, best_mae, device)
        expected_rmse, mae = test(gatRec, device, test_loader)
        # please add the validation set to tune the hyper-parameters based on your datasets.
        # early stopping (no validation set in toy dataset)
        if best_rmse > expected_rmse:
            best_rmse = expected_rmse
            best_mae = mae
            endure_count = 0
        else:
            endure_count += 1
        print("rmse: %.4f, mae:%.4f " % (expected_rmse, mae))

        if endure_count > 5:
            break
    print('Finished.')


if __name__ == '__main__':
    main()

