# -*- coding: utf-8 -*-
# @Time    : 2020/4/18 14:07
# @Author  : Aurora
# @File    : run_example_onlyL1.py
# @Function: 

import argparse
import os
import sys
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt

import data_loader_new
from logger.Logger import Logger
from model.L1neighs_Aggregator import L1neighs_Aggregator
from model.Encoder_L1 import Encoder_L1
from model.GATrec import GATrec

from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")

def train(model, train_loader, optimizer, epoch, best_macro_f1, best_micro_f1, device):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        batch_start = time.time()
        batch_nodes, labels_list = data
        optimizer.zero_grad()
        loss = model.loss(batch_nodes.to(device), labels_list.to(device))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        p_i = 10
        if i % p_i == 0:
            print('[%d, %5d] loss: %.3f, The best_macro_f1/ best_micro_f1: %.6f / %.6f' % (
                epoch, i, running_loss / p_i, best_macro_f1, best_micro_f1))
            running_loss = 0.0
        batch_end = time.time()
        # print("loss:",loss)
        # print('batch train time cost:{} s'.format(batch_end-batch_start))
    return 0

def test(model, device, test_loader):
    model.eval()
    tmp_pred = []
    target = []
    with torch.no_grad():
        for test_ap, tmp_target in test_loader:
            test_ap, tmp_target = test_ap.to(device), tmp_target.to(device)
            val_output = model.forward(test_ap)
            tmp_pred.append(list(val_output.data.argmax(dim=1).cpu().numpy()))
            # target.append(list(tmp_target.data.argmax(dim=1).cpu().numpy()))
            target.append(list(tmp_target.data.cpu().numpy()))

    y_predict = np.array(sum(tmp_pred, []))
    y_test = np.array(sum(target, []))
    print("predict",y_predict[-10:])
    print("target",y_test[-10:])
    # expected_rmse = sqrt(mean_squared_error(tmp_pred, target))
    # mae = mean_absolute_error(tmp_pred, target)

    print('准确率:', metrics.accuracy_score(y_test, y_predict)) #预测准确率输出
 
    print('宏平均精确率:',metrics.precision_score(y_test,y_predict,average='macro')) #预测宏平均精确率输出
    print('微平均精确率:', metrics.precision_score(y_test, y_predict, average='micro')) #预测微平均精确率输出
    print('加权平均精确率:', metrics.precision_score(y_test, y_predict, average='weighted')) #预测加权平均精确率输出
     
    print('宏平均召回率:',metrics.recall_score(y_test,y_predict,average='macro'))#预测宏平均召回率输出
    print('微平均召回率:',metrics.recall_score(y_test,y_predict,average='micro'))#预测微平均召回率输出
    print('加权平均召回率:',metrics.recall_score(y_test,y_predict,average='micro'))#预测加权平均召回率输出
     
    macro_f1 = metrics.f1_score(y_test,y_predict,labels=[0,1,2,3,4,5,6,7],average='macro')
    print('宏平均F1-score:',macro_f1)#预测宏平均f1-score输出
    micro_f1 = metrics.f1_score(y_test,y_predict,labels=[0,1,2,3,4,5,6,7],average='micro')
    print('微平均F1-score:',micro_f1)#预测微平均f1-score输出
    print('加权平均F1-score:',metrics.f1_score(y_test,y_predict,labels=[0,1,2,3,4,5,6,7],average='weighted'))#预测加权平均f1-score输出
     
    print('混淆矩阵输出:\n',metrics.confusion_matrix(y_test,y_predict,labels=[0,1,2,3,4,5,6,7]))#混淆矩阵输出
    print('分类报告:\n', metrics.classification_report(y_test, y_predict,labels=[0,1,2,3,4,5,6,7]))#分类报告输出


    return macro_f1, micro_f1

def main():
    parser = argparse.ArgumentParser(description='weihaoGNN')
    parser.add_argument('-bs', '--batch_size', type=int, default=128, help='input batch size for training')
    # parser.add_argument('-ed', '--embed_dim', type=int, default=300, help='embedding size')
    parser.add_argument('-ed', '--embed_dim', type=int, default=64, help='embedding size')
    parser.add_argument('-lr', '--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('-tbs', '--test_batch_size', type=int, default=1000, help='input batch size for testing')
    parser.add_argument('-ep', '--epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('-c', '--count', type=int, default=100836, help='number of edges for training and testing')
    parser.add_argument('-r', '--l2rate', type=float, default=1.0, help='rate of selected l2 paths')
    parser.add_argument('-num_author', '--number_of_author', type=int, default=16604, help='number of author')
    parser.add_argument('-num_paper', '--number_of_paper', type=int, default=12455, help='number of paper')
    args = parser.parse_args()

    path_class_a = './clean_data/author_class.txt'
    path_class_p = './clean_data/paper_class.txt'

    path_movies = './clean_data/Aminer_Graph.txt'

    path_log = './log/'
    path_log_err = './log/err/'

    now = datetime.now()
    now = now.strftime('%m%d_%H%M%S')
    sys.stdout = Logger(path_log+'run_example_L1-'+now+'.log', sys.stdout)
    sys.stderr = Logger(path_log_err+'run_example_L1_err-'+now+'.log', sys.stderr)

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
    number_of_author = args.number_of_author
    number_of_paper = args.number_of_paper

    print('#### Ziheng--Aminer ####\n')
    print('parameter:\n{}'.format(args.__dict__))

    train_ap, test_ap, train_class, test_class = data_loader_new.get_train_test_withcount(path_class_a, path_class_p, 0.2)

    G_ap = data_loader_new.getGraphFromFile("./clean_data/Aminer_author2paper.txt", args.number_of_author, args.number_of_paper)
    G_aa = data_loader_new.getGraphFromFile("./clean_data/Aminer_coauthor.txt", args.number_of_author, number_of_author)
    G_pp = data_loader_new.getGraphFromFile("./clean_data/Aminer_citation.txt", args.number_of_paper, args.number_of_paper)

    ap_L1path = data_loader_new.getL1paths(G_ap, ap="ap")
    # print(ap_L1path)
    pa_L1path = data_loader_new.getL1paths(G_ap, ap="pa")
    # print(pa_L1path)
    aa_L1path = data_loader_new.getL1paths(G_aa, ap="aa")
    # print(aa_L1path)
    pp_L1path = data_loader_new.getL1paths(G_pp, ap="pp")
    # print(pp_L1path)

    trainset = torch.utils.data.TensorDataset(torch.LongTensor(train_ap), torch.FloatTensor(train_class))
    testset = torch.utils.data.TensorDataset(torch.LongTensor(test_ap), torch.FloatTensor(test_class))
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=True)

    a2e = data_loader_new.get_ap_feature("./clean_data/author_vectors.txt")
    p2e = data_loader_new.get_ap_feature("./clean_data/paper_vectors.txt")

    # a2e = nn.Embedding(number_of_author, embed_dim).to(device).weight
    # p2e = nn.Embedding(number_of_paper, embed_dim).to(device).weight

    ap_L1Aggregator = L1neighs_Aggregator(a2e, p2e, "ap", embed_dim, cuda=device)
    pa_L1Aggregator = L1neighs_Aggregator(a2e, p2e, "pa", embed_dim, cuda=device)
    aa_L1Aggregator = L1neighs_Aggregator(a2e, p2e, "aa", embed_dim, cuda=device)
    pp_L1Aggregator = L1neighs_Aggregator(a2e, p2e, "pp", embed_dim, cuda=device)

    # u_Encoder = Encoder_L1(a2e, embed_dim, uL1paths, u_L1Aggregator, cuda=device)
    # v_Encoder = Encoder_L1(p2e, embed_dim, vL1paths, v_L1Aggregator, cuda=device)
    ap_Encoder = Encoder_L1(a2e, p2e, embed_dim, ap_L1path, pa_L1path, aa_L1path, pp_L1path, \
                        ap_L1Aggregator, pa_L1Aggregator, aa_L1Aggregator, pp_L1Aggregator, cuda=device)

    gatRec = GATrec(ap_Encoder, embed_dim).to(device)
    # optimizer = torch.optim.RMSprop(gatRec.parameters(), lr=lr, alpha=0.99)
    optimizer = torch.optim.Adam(gatRec.parameters(), lr=lr)

    best_macro_f1 = 0
    best_micro_f1 = 0
    endure_count = 0

    for epoch in range(1, epochs + 1):
        train(gatRec, train_loader, optimizer, epoch, best_macro_f1, best_micro_f1, device)
        macro_f1, micro_f1 = test(gatRec, device, test_loader)
        # please add the validation set to tune the hyper-parameters based on your datasets.
        # early stopping (no validation set in toy dataset)
        if best_micro_f1 < micro_f1:
            best_macro_f1 = macro_f1
            best_micro_f1 = micro_f1
            endure_count = 0
        else:
            endure_count += 1
        print("macro_f1: %.4f, micro_f1:%.4f " % (macro_f1, micro_f1))

        if endure_count > 5:
            break
    print('Finished.')


if __name__ == '__main__':
    main()