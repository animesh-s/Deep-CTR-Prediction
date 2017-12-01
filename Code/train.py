# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 22:42:01 2017

@author: Archit
"""

import bz2
import datetime
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import math
import model
import os
import pickle
import time
import torch
import torch.nn as nn
import warnings
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score
import pdb

warnings.filterwarnings("ignore")


dates = ['201310' + str(i) for i in range(19, 28)]
alldicts_filepath = "../Processed Data/alldicts.pkl"
#clkbidids, key2ind, set_keys, dict_list = pickle.load(open(alldicts_filepath, "rb"))
dicts = pickle.load(open(alldicts_filepath, "rb"))


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def showPlot(points, save_path):
    plt.figure()
    plt.plot(points)
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('NLL Loss', fontsize=12)
    plt.title('NLL Loss vs Number of iterations')
    plt.grid()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def variable(x):
    return Variable(torch.LongTensor([x]))


def train(args, model, lr, weight_decay):
    pos_count, neg_count = 0, 0
    start = time.time()
    plot_losses = []
    print_loss_total = 0    # Reset every args.log_interval
    plot_loss_total = 0     # Reset every args.plot_interval
    model_optimizer = torch.optim.SGD(model.parameters(), lr = lr, weight_decay = weight_decay)
    weight = torch.Tensor([[args.imbalance_factor, 1]])
    criterion = nn.CrossEntropyLoss(weight) #, ignore_index = 0)
    iter = 1
    seen_bidids = set()
    while iter < args.epochs:
        print('iteration number:', iter)
        for date in dates:     
            filepath = '../Data/training3rd/imp.' + date + '.txt.bz2'
            with bz2.BZ2File(filepath) as f:
                for line in f:
                    line = line.split('\n')[0].split('\t')
                    true_label = 1 if line[dicts[1]['bidid']] in dicts[0][0] else 0
                    if (pos_count == 0 or float(neg_count) / pos_count > args.imbalance_factor) and true_label == 0:
                        continue
                    elif true_label == 0:
                        neg_count += 1
                    else:
                        pos_count += 1
                    seen_bidids.add(line[dicts[1]['bidid']])
                    model_optimizer.zero_grad()
                    output = model(line, dicts)
                    loss = criterion(output, variable(true_label))
                    loss.backward()
                    model_optimizer.step()
                    print_loss_total += loss.data[0]
                    plot_loss_total += loss.data[0]
                    if iter % args.log_interval == 0:
                        print_loss_avg = print_loss_total / args.log_interval
                        print_loss_total = 0
                        print('%s (%d %d%%) %.10f' % (timeSince(start, float(iter) / args.epochs),
                                   iter, float(iter) / args.epochs * 100, print_loss_avg))
                    if iter % args.plot_interval == 0:
                        plot_loss_avg = plot_loss_total / args.plot_interval
                        plot_losses.append(plot_loss_avg)
                        plot_loss_total = 0
                    if iter % args.save_interval == 0:
                        if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir)
                        lr_save_dir = os.path.join(args.save_dir, 'lr_' + str(lr))
                        if not os.path.isdir(lr_save_dir): os.makedirs(lr_save_dir)
                        save_prefix = os.path.join(lr_save_dir, 'model_' + str(weight_decay))
                        save_path = '{}_steps{}.pt'.format(save_prefix, iter)
                        torch.save(model, save_path)
                    if iter == args.epochs:
                        break
                    iter += 1
    print(pos_count, neg_count)
    if not os.path.isdir(args.plot_dir): os.makedirs(args.plot_dir)
    prefix = 'lr_' + str(lr) + '.png'
    save_path = os.path.join(args.plot_dir, prefix)
    showPlot(plot_losses, save_path)
    return model, seen_bidids


def cross_validation(args):
    for learning_rate in args.lr:
        print('Learning Rate:', learning_rate)
        for weight_decay in args.weight_decay:
            print('Weight Decay: ', weight_decay)
            for factor in args.factors:
                print('Factor: ', factor)
                args.factor = factor
                LRmodel = model.LR(args)
                LRmodel, seen_bidids = train(args, LRmodel, learning_rate, weight_decay)
                correct, wrong, accuracy, auc = evaluate(args, LRmodel, seen_bidids)
                print 'Correct: ' + str(correct) + ' Wrong: ' + str(wrong) + ' Accuracy: ' + str(accuracy) + ' AUC: ' + str(auc)


def evaluate(args, model, seen_bidids):
    pos_count, neg_count = 0, 0
    correct, wrong = 0, 0
    true_labels, predicted_labels = [], []
    for date in dates:     
        filepath = '../Data/training3rd/imp.' + date + '.txt.bz2'
        with bz2.BZ2File(filepath) as f:
            for line in f:
                line = line.split('\n')[0].split('\t')
                if line[dicts[1]['bidid']] in seen_bidids:
                    continue
                true_label = 1 if line[dicts[1]['bidid']] in dicts[0][1] else 0
                if (pos_count == 0 or float(neg_count) / pos_count > args.imbalance_factor) and true_label == 0:
                    continue
                elif true_label == 0:
                    neg_count += 1
                else:
                    pos_count += 1
                true_labels.append(true_label)
                output = model(line, dicts)
                predicted_label = 0 if output.data[0][0] >= output.data[0][1] else 1
                predicted_labels.append(predicted_label)
                if predicted_label == true_label:
                    correct += 1
                else:
                    wrong += 1
    return correct, wrong, float(correct) / (correct + wrong), roc_auc_score(true_labels, predicted_labels)


def evaluatefull(model):
    correct, wrong = 0, 0
    true_labels, predicted_labels = [], []
    iter = 1
    for date in dates:
        print(date, correct, wrong)
        filepath = '../Data/training3rd/imp.' + date + '.txt.bz2'
        with bz2.BZ2File(filepath) as f:
            for line in f:
                print(iter)
                line = line.split('\n')[0].split('\t')
                true_label = 1 if line[dicts[1]['bidid']] in dicts[0] else 0
                true_labels.append(true_label)
                output = model(line, dicts)
                predicted_label = 0 if output.data[0][0] >= output.data[0][1] else 1
                predicted_labels.append(predicted_label)
                if predicted_label == true_label:
                    correct += 1
                else:
                    wrong += 1
                iter += 1
    return correct, wrong, float(correct) / (correct + wrong), roc_auc_score(true_labels, predicted_labels)


