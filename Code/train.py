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
import warnings
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve

warnings.filterwarnings("ignore")


dates = ['201310' + str(i) for i in range(19, 28)]
alldicts_filepath = "../Processed Data/alldicts.pkl"
dicts = pickle.load(open(alldicts_filepath, "rb")) #clkbidids, key2ind, set_keys, dict_list


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
    print_loss_total = 0
    plot_loss_total = 0
    if args.optimizer == 'SGD':
        model_optimizer = torch.optim.SGD(
                model.parameters(), lr = args.lr,
                weight_decay = args.weight_decay)
    elif args.optimizer == 'Adam':
        model_optimizer = torch.optim.Adam(
                model.parameters(), lr = args.lr,
                weight_decay = args.weight_decay)
    iter, epoch = 0, 1
    train_seen_bidids = set()
    batch_loss = Variable(torch.FloatTensor([0]))
    while iter < args.iterations:
        print('epoch:', epoch)
        epoch += 1
        np.random.shuffle(dates)
        for date in dates:
            if iter == args.iterations: 
                break
            filepath = '../Data/training3rd/imp.' + date + '.txt.bz2'
            with bz2.BZ2File(filepath) as f:
                for line in f:
                    line = line.split('\n')[0].split('\t')
                    if line[dicts[1]['bidid']] in dicts[0][1]\
                        or line[dicts[1]['bidid']] in dicts[0][2]:
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
                    train_seen_bidids.add(line[dicts[1]['bidid']])
                    output = model(line, dicts)
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
                        print('%s (%d %d%%) %.10f' % (
                                timeSince(start, float(iter) / args.iterations),
                                iter, float(iter) / args.iterations * 100,
                                print_loss_avg))
                    if iter % args.plot_interval == 0:
                        plot_loss_avg = plot_loss_total / args.plot_interval
                        plot_losses.append(plot_loss_avg)
                        plot_loss_total = 0
                    if iter == args.iterations:
                        break
    print('pos_count:', pos_count, 'neg_count:', neg_count)
    if not args.cv:
        if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir)
        save_path = os.path.join(args.save_dir, 'model.pt')
        torch.save(model, save_path)
        if not os.path.isdir(args.plot_dir): os.makedirs(args.plot_dir)
        save_path = os.path.join(args.plot_dir, 'learning_curve.png')
        showPlot(plot_losses, save_path)
    return train_seen_bidids
        

def cross_validation(args):
    for iter in range(1, args.num_models + 1):
        args.lr = 10**np.random.uniform(-5, -1)
        args.weight_decay = 10**np.random.uniform(-5, 1)
        if args.modeltype in ['LR', 'MLP']:
            args.factor = np.random.randint(90, 111)
            model = models.LR(args) if args.modeltype == 'LR'\
                else models.MLP(args)
            print('{}, Model: {}, lr: {:.5f}, wd: {:.5f}, factor: {}'
                  .format(iter, args.modeltype, args.lr, args.weight_decay,
                          args.factor))
            f = open(args.filepath, 'a')
            f.write('%d, Model: %s, lr: %.5f, wd: %.5f, factor: %d\n' %(
                    iter, args.modeltype, args.lr, args.weight_decay,
                    args.factor))
            f.close()
        elif args.modeltype in ['CNN', 'CNNDeep']:
            args.embed_dim = np.random.randint(90, 111)
            args.kernel_num = np.random.randint(90, 111)
            args.dropout = np.random.uniform(0.1, 0.5)
            model = models.CNN(args) if args.modeltype == 'CNN'\
                else models.CNNDeep(args)
            print('{}, Model: {}, lr: {:.5f}, wd: {:.5f}, embed_dim: {},\
                  kernel_num: {}, dropout: {:.5f}'.format(iter, args.modeltype,
                  args.lr, args.weight_decay, args.embed_dim, args.kernel_num,
                  args.dropout))
            f = open(args.filepath, 'a')
            f.write('%d, Model: %s, lr: %.5f, wd: %.5f, embed_dim: %d,\
                    kernel_num: %d, dropout: %.5f\n' %(iter, args.modeltype,
                    args.lr, args.weight_decay, args.embed_dim, args.kernel_num,
                    args.dropout))
            f.close()
        train_and_evaluate(args, model)


def train_and_evaluate(args, model):
    train_seen_bidids = train(args, model)
    f1, auc, _ = evaluate(args, model, train_seen_bidids, train = True)
    print('Training f1: {:.5f}, AUC: {:.5f}'.format(f1, auc))
    f = open(args.filepath, 'a')
    f.write('Training f1: %.5f, AUC: %.5f\n' %(f1, auc))
    f.close()
    f1, auc, valid_seen_bidids = evaluate(args, model, train_seen_bidids,
                                          valid = True)
    print('Validation f1: {:.5f}, AUC: {:.5f}'.format(f1, auc))
    f = open(args.filepath, 'a')
    f.write('Validation f1: %.5f, AUC: %.5f\n' %(f1, auc))
    f.close()
    f1, auc, _ = evaluate(args, model, train_seen_bidids, valid_seen_bidids,
                          test = True)
    if not args.cv:
        print('Test f1: {:.5f}, AUC: {:.5f}'.format(f1, auc))
        f = open(args.filepath, 'a')
        f.write('Test f1: %.5f, AUC: %.5f\n' %(f1, auc))
        f.close()


def evaluate(args, model, train_seen_bidids, valid_seen_bidids = set(),
             train = False, valid = False, test = False):
    pos_count, neg_count = 0, 0
    true_labels, predicted_labels, predicted_scores = [], [], []
    for date in dates:     
        filepath = '../Data/training3rd/imp.' + date + '.txt.bz2'
        with bz2.BZ2File(filepath) as f:
            for line in f:
                line = line.split('\n')[0].split('\t')
                if train:
                    if line[dicts[1]['bidid']] not in train_seen_bidids:
                        continue
                    true_label = 1 if line[dicts[1]['bidid']] in dicts[0][0] else 0
                else:
                    if valid:
                        if line[dicts[1]['bidid']] in train_seen_bidids:
                            continue
                        true_label = 1 if line[dicts[1]['bidid']] in dicts[0][1]\
                            else 0
                    elif test:
                        if line[dicts[1]['bidid']] in train_seen_bidids\
                            or line[dicts[1]['bidid']] in valid_seen_bidids:
                            continue
                        true_label = 1 if line[dicts[1]['bidid']] in dicts[0][2]\
                            else 0
                    if (pos_count == 0 \
                        or float(neg_count) / pos_count > args.imbalance_factor)\
                        and true_label == 0:
                        continue
                    elif true_label == 0:
                        neg_count += 1
                    else:
                        pos_count += 1
                    if valid:
                        valid_seen_bidids.add(line[dicts[1]['bidid']])
                true_labels.append(true_label)
                output = model(line, dicts, infer = True)
                predicted_scores.append(output.data[0][1])
                predicted_label = 0 if output.data[0][0] > args.threshold else 1
                predicted_labels.append(predicted_label)
    confusion_mat = confusion_matrix(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels, average='macro')
    if test:
        fpr, tpr, _ = roc_curve(true_labels, predicted_scores)
        f = open(args.filepath, 'a')
        f.write('fpr\n')
        for i in fpr:
            f.write('%.5f,' %(i))
        f.write('\ntpr\n')
        for i in tpr:
            f.write('%.5f,' %(i))
        f.write('\n')
        f.close()
    return f1, roc_auc_score(true_labels, predicted_scores), valid_seen_bidids


def final_run(args):
    if args.modeltype == 'LR':
        model = models.LR(args)
    elif args.modeltype == 'MLP':
        model = models.MLP(args)
    elif args.modeltype == 'CNN':
        model = models.CNN(args)
    elif args.modeltype == 'CNNDeep':
        model = models.CNNDeep(args)
    train_and_evaluate(args, model)
