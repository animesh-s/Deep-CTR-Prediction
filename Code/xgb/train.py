import bz2
import sys
import numpy as np
sys.path.append('..')
import model as models
import os
import pickle
import warnings
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import xgboost as xgb

warnings.filterwarnings("ignore")


dates = ['201310' + str(i) for i in range(19, 28)]
alldicts_filepath = "../../Processed Data/alldicts.pkl"
#clkbidids, key2ind, set_keys, dict_list
dicts = pickle.load(open(alldicts_filepath, "rb"))


def train(args, Xgmodel, AEmodel):
    pos_count, neg_count = 0, 0
    training_samples, training_labels = [], []
    iter = 1
    train_seen_bidids = set()
    while iter < args.iterations:
        print('iteration number:', iter)
        for date in dates:     
            filepath = '../../Data/training3rd/imp.' + date + '.txt.bz2'
            with bz2.BZ2File(filepath) as f:
                for line in f:
                    line = line.split('\n')[0].split('\t')
                    if line[dicts[1]['bidid']] in dicts[0][1]\
                        or line[dicts[1]['bidid']] in dicts[0][2]:
                        continue
                    true_label = 1 if line[dicts[1]['bidid']] in dicts[0][0] else 0
                    if (pos_count == 0 or float(neg_count) / pos_count > args.imbalance_factor) and true_label == 0:
                        continue
                    elif true_label == 0:
                        neg_count += 1
                    else:
                        pos_count += 1
                    train_seen_bidids.add(line[dicts[1]['bidid']])
                    training_sample = Xgmodel(line, dicts)
                    training_samples.append(training_sample)
                    training_labels.append(true_label)
                    if iter == args.iterations:
                        break
                    iter += 1
    dtrain = xgb.DMatrix(training_samples, training_labels)
    param = {'max_depth': args.max_depth, 'eta': args.lr, 'silent': 1, 'objective': 'binary:logistic'}
    bst = xgb.train(param, dtrain, args.num_rounds)
    print('pos_count:', pos_count, 'neg_count:', neg_count)
    if not args.cv:
        if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir)
        save_path = os.path.join(args.save_dir, 'xgboost.model')
        bst.save_model(save_path)
    return bst, train_seen_bidids

def cross_validation(args):
    for iter in range(1, args.num_models + 1):
        args.lr = 10**np.random.uniform(-5, -1)
        args.max_depth = np.random.randint(1, 31)
        args.num_rounds = np.random.randint(10, 71)
        args.factor = np.random.randint(90, 111)
        print('{}, Model: {}, lr: {:.5f}, max_depth: {}, num_rounds: {},'\
              ' ae_lr: {:.5f}, wd: {:.5f}, factor: {}'.format(iter,
              args.modeltype, args.lr, args.max_depth, args.num_rounds,
              args.ae_lr, args.weight_decay, args.factor))
        f = open(args.filepath, 'a')
        f.write('%d, Model: %s, lr: %.5f, max_depth: %d, num_rounds: %d,'\
                ' ae_lr: %.5f, wd: %.5f, factor: %d\n' %(iter, args.modeltype,
                args.lr, args.max_depth, args.num_rounds, args.ae_lr,
                args.weight_decay, args.factor))
        f.close()
        train_and_evaluate(args)


def train_and_evaluate(args):
    Xgbmodel = models.Xgb(args)
    AEmodel = models.Autoencoder(args)
    bst, train_seen_bidids = train(args, Xgbmodel, AEmodel)
    auc, valid_seen_bidids = evaluate(args, Xgbmodel, AEmodel, bst,
                                      train_seen_bidids)
    print('Validation AUC: {:.5f}'.format(auc))
    f = open(args.filepath, 'a')
    f.write('Validation AUC: %.5f\n' %(auc))
    f.close()
    if not args.cv:
        auc, _ = evaluate(args, Xgbmodel, AEmodel, bst, train_seen_bidids,
                          valid_seen_bidids, False)
        print('Test AUC: {:.5f}'.format(auc))
        f = open(args.filepath, 'a')
        f.write('Test AUC: %.5f\n' %(auc))
        f.close()


def final_run(args):
    train_and_evaluate(args)


def evaluate(args, Xgbmodel, AEmodel, bst, train_seen_bidids, valid_seen_bidids = set(), valid = True):
    pos_count, neg_count = 0, 0
    true_labels, predicted_labels = [], []
    test_samples = []
    for date in dates:     
        filepath = '../../Data/training3rd/imp.' + date + '.txt.bz2'
        with bz2.BZ2File(filepath) as f:
            for line in f:
                line = line.split('\n')[0].split('\t')
                if valid:
                    if line[dicts[1]['bidid']] in train_seen_bidids:
                        continue
                    true_label = 1 if line[dicts[1]['bidid']] in dicts[0][1] else 0
                else:
                    if line[dicts[1]['bidid']] in train_seen_bidids\
                        or line[dicts[1]['bidid']] in valid_seen_bidids:
                        continue
                    true_label = 1 if line[dicts[1]['bidid']] in dicts[0][2] else 0
                if (pos_count == 0 or float(neg_count) / pos_count > args.imbalance_factor) and true_label == 0:
                    continue
                elif true_label == 0:
                    neg_count += 1
                else:
                    pos_count += 1
                if valid:
                    valid_seen_bidids.add(line[dicts[1]['bidid']])
                true_labels.append(true_label)
                test_sample = Xgbmodel(line, dicts)
                test_samples.append(test_sample)
    dtest = xgb.DMatrix(test_samples)
    predicted_labels = bst.predict(dtest)
    if not valid:
        fpr, tpr, _ = roc_curve(true_labels, predicted_labels)
        f = open(args.filepath, 'a')
        f.write('fpr\n')
        for i in fpr:
            f.write('%.5f,' %(i))
        f.write('\ntpr\n')
        for i in tpr:
            f.write('%.5f,' %(i))
        f.write('\n')
        f.close()
    return roc_auc_score(true_labels, predicted_labels), valid_seen_bidids
