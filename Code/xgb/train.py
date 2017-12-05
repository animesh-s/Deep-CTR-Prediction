import bz2
import datetime
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import math
import sys
import numpy as np
sys.path.append('..')
import model as models
import os
import pickle
import time
import warnings
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import torch
from torch import nn
from torch.autograd import Variable
import pdb

warnings.filterwarnings("ignore")


dates = ['201310' + str(i) for i in range(19, 28)]
alldicts_filepath = "../../Processed Data/alldicts.pkl"
#clkbidids, key2ind, set_keys, dict_list = pickle.load(open(alldicts_filepath, "rb"))
dicts = pickle.load(open(alldicts_filepath, "rb"))


def train(args, Xgmodel, AEmodel):
    pos_count, neg_count = 0, 0
    start = time.time()
    training_samples, training_labels = [], []
    training_samples_encoded = []
    iter = 1
    seen_bidids = set()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(AEmodel.parameters(), lr=args.ae_lr, weight_decay=args.weight_decay)
    while iter < args.epochs:
        print('iteration number:', iter)
        for date in dates:     
            filepath = '../../Data/training3rd/imp.' + date + '.txt.bz2'
            with bz2.BZ2File(filepath) as f:
                for line in f:
                    line = line.split('\n')[0].split('\t')
                    if line[dicts[1]['bidid']] in dicts[0][1]:
                        continue
                    true_label = 1 if line[dicts[1]['bidid']] in dicts[0][0] else 0
                    if (pos_count == 0 or float(neg_count) / pos_count > args.imbalance_factor) and true_label == 0:
                        continue
                    elif true_label == 0:
                        neg_count += 1
                    else:
                        pos_count += 1
                    seen_bidids.add(line[dicts[1]['bidid']])
                    training_sample = Xgmodel(line, dicts)
                    training_sample = Variable(torch.FloatTensor(training_sample)).view(1,-1)
                    training_samples.append(training_sample)
                    encoded_output = AEmodel.encode(training_sample)
                    output = AEmodel.decode(encoded_output)
                    loss = criterion(output, training_sample)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    training_labels.append(true_label)
                    if iter == args.epochs:
                        break
                    iter += 1
    for training_sample in training_samples:
        training_sample = AEmodel.encode(training_sample).data[0].numpy()
        training_samples_encoded.append(training_sample)
    dtrain = xgb.DMatrix(training_samples_encoded, training_labels)
    param = {'max_depth': args.max_depth, 'eta': args.lr, 'silent': 1, 'objective': 'multi:softmax', 'num_class': 2}
    bst = xgb.train(param, dtrain, args.num_rounds)
    print('pos_count:', pos_count, 'neg_count:', neg_count)
    return bst, seen_bidids

def cross_validation(args):
    for iter in range(1, args.num_models + 1):
        args.lr = 10**np.random.uniform(-5, -1)
        args.max_depth = int(np.random.uniform(30, 100))
        args.num_rounds = int(np.random.uniform(10, 70))
        args.ae_lr = 10**np.random.uniform(-5, -1)
        args.weight_decay = 10**np.random.uniform(-5, -1)
        args.factor = int(np.random.uniform(90, 110))
        epoch = 5
        args.epochs = epoch * (args.imbalance_factor + 1) * 2160
        print('{}, Model: {}, epochs: {}, lr: {:.5f}, max_depth: {}, num_rounds: {}, ae_lr: {:.5f}, wd: {:.5f}, factor: {}'.format(
                iter, args.modeltype, epoch, args.lr, args.max_depth, args.num_rounds, args.ae_lr, args.weight_decay, args.factor))
        f = open(args.filepath, 'a')
        f.write('%d, Model: %s, epochs: %d, lr: %.5f, max_depth: %d, num_rounds: %d, ae_lr: %.5f, wd: %.5f, factor: %d\n' %(
                iter, args.modeltype, epoch, args.lr, args.max_depth, args.num_rounds, args.ae_lr, args.weight_decay, args.factor))
        f.close()
        Xgbmodel = models.Xgb(args)
        AEmodel = models.Autoencoder(args)
        bst, seen_bidids = train(args, Xgbmodel, AEmodel)
        correct, wrong, accuracy, auc = evaluate(args, Xgbmodel, AEmodel, bst, seen_bidids)
        print('Validation Correct: {}, Wrong: {}, Accuracy: {:.5f}, AUC: {:.5f}'
              .format(correct, wrong, accuracy, auc))
        f = open(args.filepath, 'a')
        f.write('Validation Correct: %d, Wrong: %d, Accuracy: %.5f, AUC: %.5f\n' 
                %(correct, wrong, accuracy, auc))
        f.close()

def evaluate(args, Xgbmodel, AEmodel, bst, seen_bidids):
    pos_count, neg_count = 0, 0
    correct, wrong = 0, 0
    true_labels, predicted_labels = [], []
    test_samples = []
    for date in dates:     
        filepath = '../../Data/training3rd/imp.' + date + '.txt.bz2'
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
                test_sample = Xgbmodel(line, dicts)
                test_sample = Variable(torch.FloatTensor(test_sample)).view(1,-1)
                test_sample = AEmodel.encode(test_sample).data[0].numpy()
                test_samples.append(test_sample)
    dtest = xgb.DMatrix(test_samples)
    predicted_labels = bst.predict(dtest)
    for true_label, predicted_label in zip(true_labels, predicted_labels):
        if true_label == predicted_label:
            correct += 1
        else:
            wrong += 1
    return correct, wrong, float(correct) / (correct + wrong), roc_auc_score(true_labels, predicted_labels)
