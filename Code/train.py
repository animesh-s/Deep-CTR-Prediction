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
import matplotlib.ticker as ticker
import math
import os
import pickle
import time
import torch
import torch.nn as nn
import warnings
from torch.autograd import Variable
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


def showPlot(points):
    plt.figure()
    plt.plot(points)
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('NLL Loss', fontsize=12)
    plt.title('NLL Loss vs Number of iterations')
    plt.grid()
    date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    plt.savefig('training_curve_' + date, bbox_inches='tight')
    plt.close()


def variable(x):
    return Variable(torch.LongTensor([x]))


def train(args, model):
    pos_count, neg_count = 0, 0
    start = time.time()
    plot_losses = []
    print_loss_total = 0    # Reset every args.log_interval
    plot_loss_total = 0     # Reset every args.plot_interval
    model_optimizer = torch.optim.SGD(model.parameters(), lr = args.lr)
    criterion = nn.NLLLoss()
    
    iter = 1
    
    while iter < args.epochs:
        print(iter)
        for date in dates:     
            filepath = '../Data/training3rd/imp.' + date + '.txt.bz2'
            with bz2.BZ2File(filepath) as f:
                for line in f:
                    line = line.split('\n')[0].split('\t')
                    true_label = 1 if line[dicts[1]['bidid']] in dicts[0] else 0
                    
                    if neg_count > pos_count and true_label == 0:
                        continue
                    elif true_label == 0:
                        neg_count += 1
                    else:
                        pos_count += 1    
                    
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
                        print('%s (%d %d%%) %.4f' % (timeSince(start, float(iter) / args.epochs),
                                   iter, float(iter) / args.epochs * 100, print_loss_avg))
                    if iter % args.plot_interval == 0:
                        plot_loss_avg = plot_loss_total / args.plot_interval
                        plot_losses.append(plot_loss_avg)
                        plot_loss_total = 0
                    if iter % args.save_interval == 0:
                        if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir)
                        models = [model]
                        prefixes = ['model']
                        for model, prefix in zip(models, prefixes):
                            save_prefix = os.path.join(args.save_dir, prefix)
                            save_path = '{}_steps{}.pt'.format(save_prefix, iter)
                            torch.save(model, save_path)
                    if iter == args.epochs:
                        break
                
                    iter += 1

    print(pos_count, neg_count)
    showPlot(plot_losses)