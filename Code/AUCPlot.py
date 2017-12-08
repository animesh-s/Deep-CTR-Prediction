# -*- coding: utf-8 -*-
"""
Created on Thu Dec 07 19:20:44 2017

@author: Archit
"""

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import glob
from itertools import cycle
import warnings
from sklearn.metrics import auc

warnings.filterwarnings("ignore")


color = ['cyan', 'indigo', 'seagreen', 'darkorange', 'yellow']
lw = 2


def plotAUC(files, colors, classifiers, identifier):
    plt.figure()
    for  f, color, classifier in zip(files, colors, classifiers):
        fpr,tpr = read_data(f)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=lw, color=color,
             label='%s (%0.3f)' %(classifier, roc_auc))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    a = identifier.split('_')
    plt.title('Optimizer: {}, Embedding: {}, Imbalance Ratio: 1:{}'
              .format(a[0], a[1], a[2]))
    plt.legend(loc="lower right")
    plt.savefig('../AUCPlots/{}.png'.format(identifier), bbox_inches='tight')
    plt.close()


def read_data(filename):
    prev_line = ""
    with open(filename) as f:
        for line in f:
            if prev_line == "fpr\n":
                line = line.split(',')[:-1]
                fpr = [float(x) for x in line]
            if prev_line == "tpr\n":
                line = line.split(',')[:-1]
                tpr = [float(x) for x in line]
            prev_line = line
    return fpr, tpr


if __name__ == '__main__':

    embeddings = ['Static', 'Dynamic']
    optimizers = ['SGD', 'Adam']
    imbalances = ['9', '49']

    for embedding in embeddings:
        for optimizer in optimizers:
            for imbalance in imbalances:
                identifier = '{}_{}_{}'.format(embedding, optimizer, imbalance)
                files_location = '../Snapshots/{}/*.txt'.format(identifier)
                files = glob.glob(files_location)
                num_files = len(files)
                colors = cycle(color[:num_files])
                classifiers = [filename.split('\\')[1].split('_')[0] \
                                   for filename in files]
                plotAUC(files, colors, classifiers, identifier)
