import bz2
import datetime
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import math
import sys
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


def train(args, Xgmodel, AEmodel, lr, ae_lr, weight_decay, max_depth, num_round):
    pos_count, neg_count = 0, 0
    start = time.time()
    training_samples, training_labels = [], []
    training_samples_encoded = []
    iter = 1
    seen_bidids = set()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(AEmodel.parameters(), lr=ae_lr, weight_decay=weight_decay)
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
                    # pdb.set_trace()
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
    param = {'max_depth': max_depth, 'eta': lr, 'silent': 1, 'objective': 'multi:softmax', 'num_class': 2}
    bst = xgb.train(param, dtrain, num_round)
    print('pos_count:', pos_count, 'neg_count:', neg_count)
    return bst, seen_bidids


def cross_validation(args):
    for num_round in args.num_rounds:
        print('Num Rounds: ', num_round)
        for learning_rate in args.lr:
            print('Learning Rate:', learning_rate)
            for max_depth in args.max_depth:
                print('Max Depth: ', max_depth)
                for ae_learning_rate in args.ae_lr:
                    print('AE Learning Rate: ', ae_learning_rate)
                    for weight_decay in args.weight_decay:
                        print('Weight Decay: ', weight_decay)
                        for factor in args.factors:
                            print('Factor: ', factor)
                            args.factor = factor
                            Xgbmodel = models.Xgb(args)
                            AEmodel = models.Autoencoder(args)
                            bst, seen_bidids = train(args, Xgbmodel, AEmodel, learning_rate, ae_learning_rate, weight_decay, max_depth, num_round)
                            correct, wrong, accuracy, auc = evaluate(args, Xgbmodel, AEmodel, bst, seen_bidids)
                            print 'Correct: ' + str(correct) + ' Wrong: ' + str(wrong) + ' Accuracy: ' + str(accuracy) + ' AUC: ' + str(auc)


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