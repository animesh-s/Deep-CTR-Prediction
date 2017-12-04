# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 22:42:01 2017

@author: Archit
"""

import bz2
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import math
import model as models
import numpy as np
import os
import pickle
import time
import torch
import torch.nn as nn
import warnings
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

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


def train(args, model):
    pos_count, neg_count = 0, 0
    start = time.time()
    plot_losses = []
    print_loss_total = 0    # Reset every args.log_interval
    plot_loss_total = 0     # Reset every args.plot_interval
    model_optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, 
                                      weight_decay = args.weight_decay)
    #weight = torch.Tensor([[args.imbalance_factor, 1]])
    #criterion = nn.CrossEntropyLoss(weight) #, ignore_index = 0)
    iter, epoch = 0, 1
    seen_bidids = set()
    batch_loss = Variable(torch.FloatTensor([0]))
    while iter < args.epochs:
        print('epoch:', epoch)
        epoch += 1
        np.random.shuffle(dates)
        for date in dates:
            if iter == args.epochs: 
                break
            filepath = '../Data/training3rd/imp.' + date + '.txt.bz2'
            with bz2.BZ2File(filepath) as f:
                for line in f:
                    line = line.split('\n')[0].split('\t')
                    if line[dicts[1]['bidid']] in dicts[0][1]:
                        continue
                    true_label = 1 if line[dicts[1]['bidid']] in dicts[0][0] else 0
                    if (pos_count == 0 \
                            or float(neg_count) / pos_count > args.imbalance_factor) \
                            and true_label == 0:
                        continue
                    elif true_label == 0:
                        neg_count += 1
                    else:
                        pos_count += 1
                    seen_bidids.add(line[dicts[1]['bidid']])
                    output = model(line, dicts)
                    #loss = criterion(output, variable(true_label))
                    predicted_label = 0 if output.data[0][0] > args.threshold \
                        else 1
                    if true_label == 0:
                        loss = (1 - output[0][0])
                    else:
                        if output.data[0][0] > args.threshold:
                            loss = output[0][0] * args.imbalance_factor
                        else:
                            loss = Variable(torch.FloatTensor([0]))
                    batch_loss += loss
                    print_loss_total += loss.data[0]
                    plot_loss_total += loss.data[0]
                    iter += 1
                    if iter % args.batch_size == 0:
                        batch_loss /= args.batch_size
                        batch_loss.backward()
                        model_optimizer.step()
                        batch_loss = Variable(torch.FloatTensor([0]))
                        model_optimizer.zero_grad()
                    if iter % args.log_interval == 0:
                        print_loss_avg = print_loss_total / args.log_interval
                        print_loss_total = 0
                        print('Prob(-) %.3f true %d pred %d loss %.5f' % (
                                output.data[0][0], true_label, predicted_label,
                                loss.data[0]))
                        print('%s (%d %d%%) %.10f' % (
                                timeSince(start, float(iter) / args.epochs),
                                iter, float(iter) / args.epochs * 100,
                                print_loss_avg))
                        """
                        f = open(filename, 'a')
                        f.write('%s (%d %d%%) %.10f\n' %(
                                timeSince(start, float(iter) / args.epochs),
                                iter, float(iter) / args.epochs * 100,
                                print_loss_avg))
                        f.close()
                        """
                    if iter % args.plot_interval == 0:
                        plot_loss_avg = plot_loss_total / args.plot_interval
                        plot_losses.append(plot_loss_avg)
                        plot_loss_total = 0
                    if iter % args.save_interval == 0:
                        if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir)
                        lr_save_dir = os.path.join(args.save_dir, 'lr_' + str(args.lr))
                        if not os.path.isdir(lr_save_dir): os.makedirs(lr_save_dir)
                        save_prefix = os.path.join(lr_save_dir, 'model_' + str(args.weight_decay))
                        save_path = '{}_steps{}.pt'.format(save_prefix, iter)
                        torch.save(model, save_path)
                    if iter == args.epochs:
                        break
    print('pos_count:', pos_count, 'neg_count:', neg_count)
    """
    f = open(filename, 'a')
    f.write('pos_count: %d, neg_count: %d\n' %(pos_count, neg_count))
    f.close()
    """
    if not os.path.isdir(args.plot_dir): os.makedirs(args.plot_dir)
    prefix = 'lr_' + str(args.lr) + '.png'
    save_path = os.path.join(args.plot_dir, prefix)
    showPlot(plot_losses, save_path)
    return seen_bidids
        

def cross_validation(args):
    for iter in range(1, args.num_models + 1):
        args.lr = 10**np.random.uniform(-5, -1)
        args.weight_decay = 10**np.random.uniform(-5, 1)
        epoch = 5 #np.random.randint(1, 6)
        args.epochs = epoch * (args.imbalance_factor + 1) * 2160
        print('{}, Model: {}, epochs: {}, lr: {:.5f}, wd: {:.5f}'.format(
                iter, args.modeltype, epoch, args.lr, args.weight_decay))
        f = open(args.filepath, 'a')
        f.write('%d, Model: %s, epochs: %d, lr: %.5f, wd: %.5f\n' %(
                iter, args.modeltype, epoch, args.lr, args.weight_decay))
        f.close()
        if args.modeltype == 'LR':
            model = models.LR(args)
        elif args.modeltype == 'CNN':
            model = models.CNN(args)
        seen_bidids = train(args, model)
        correct, wrong, accuracy, auc = evaluate(args, model, seen_bidids, True)
        print('Training Correct: {}, Wrong: {}, Accuracy: {:.5f}, AUC: {:.5f}'
              .format(correct, wrong, accuracy, auc))
        f = open(args.filepath, 'a')
        f.write('Training Correct: %d, Wrong: %d, Accuracy: %.5f, AUC: %.5f\n' 
                %(correct, wrong, accuracy, auc))
        f.close()
        correct, wrong, accuracy, auc = evaluate(args, model, seen_bidids)
        print('Validation Correct: {}, Wrong: {}, Accuracy: {:.5f}, AUC: {:.5f}'
              .format(correct, wrong, accuracy, auc))
        f = open(args.filepath, 'a')
        f.write('Validation Correct: %d, Wrong: %d, Accuracy: %.5f, AUC: %.5f\n'
                %(correct, wrong, accuracy, auc))
        f.close()


def evaluate(args, model, seen_bidids, train = False):
    pos_count, neg_count = 0, 0
    correct, wrong = 0, 0
    true_labels, predicted_labels = [], []
    for date in dates:     
        filepath = '../Data/training3rd/imp.' + date + '.txt.bz2'
        with bz2.BZ2File(filepath) as f:
            for line in f:
                line = line.split('\n')[0].split('\t')
                if train:
                    if line[dicts[1]['bidid']] not in seen_bidids:
                        continue
                    true_label = 1 if line[dicts[1]['bidid']] in dicts[0][0] else 0
                else:
                    if line[dicts[1]['bidid']] in seen_bidids:
                        continue
                    true_label = 1 if line[dicts[1]['bidid']] in dicts[0][1] else 0
                    if (pos_count == 0 \
                            or float(neg_count) / pos_count > args.imbalance_factor) \
                            and true_label == 0:
                        continue
                    elif true_label == 0:
                        neg_count += 1
                    else:
                        pos_count += 1
                true_labels.append(true_label)
                output = model(line, dicts, infer = True)
                predicted_label = 0 if output.data[0][0] > args.threshold else 1
                """
                predicted_label = 0 if output.data[0][0] >= output.data[0][1] \
                    else 1
                """
                predicted_labels.append(predicted_label)
                if predicted_label == true_label:
                    correct += 1
                else:
                    wrong += 1
    confusion_mat = confusion_matrix(true_labels, predicted_labels)
    print(confusion_mat)
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    print('f1', f1)
    return correct, wrong, float(correct) / (correct + wrong), \
        roc_auc_score(true_labels, predicted_labels)


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
                output = model(line, dicts, infer = True)
                predicted_label = 0 if output.data[0][0] >= output.data[0][1] else 1
                predicted_labels.append(predicted_label)
                if predicted_label == true_label:
                    correct += 1
                else:
                    wrong += 1
                iter += 1
    return correct, wrong, float(correct) / (correct + wrong), \
        roc_auc_score(true_labels, predicted_labels)
