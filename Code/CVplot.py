# -*- coding: utf-8 -*-
"""
Created on Sat Dec 02 20:51:45 2017

@author: Archit
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

train = True
model = 'LR'

def cv1():
    auc, lr, wd = [], [], []
    line_number = 0
    with open('Snapshots/' + model + '_True.txt') as f:
        for line in f:
            line_number += 1
            a = line.strip().split(',')
            if (train and line_number % 3 == 2):
                auc.append(float(a[3].split()[1])*100)
            if (not train and line_number % 3 == 0):
                auc.append(float(a[3].split()[1])*100)
            if line_number % 3 == 1:
                lr.append(float(a[3].split()[1]))
                wd.append(float(a[4].split()[1]))
    lr = lr[:len(auc)]
    wd = wd[:len(auc)]
    auc, lr, wd = zip(*sorted(zip(auc, lr, wd)))
    auc, lr, wd = auc[-20:], lr[-20:], wd[-20:]
    return auc, lr, wd

def cv2():
    auc, epoch, factor = [], [], []
    line_number = 0
    with open('Snapshots/' + model + '_False.txt') as f:
        for line in f:
            line_number += 1
            a = line.strip().split(',')
            if (train and line_number % 3 == 2):
                auc.append(float(a[2].split()[1])*100)
            if (not train and line_number % 3 == 0):
                auc.append(float(a[2].split()[1])*100)
            if line_number % 3 == 1:
                epoch.append(float(a[2].split()[1]))
                factor.append(float(a[3].split()[1]))
    epoch = epoch[:len(auc)]
    factor = factor[:len(auc)]
    auc, epoch, factor = zip(*sorted(zip(auc, epoch, factor)))
    auc, epoch, factor = auc[-20:], epoch[-20:], factor[-20:]
    return auc, epoch, factor

auc, lr, wd = cv2()
data = pd.DataFrame(data={'LR':lr, 'wd':wd, 'z':auc})
data = data.pivot(index='wd', columns='LR', values='z')
sns.heatmap(data, annot=True, cbar=True)
#plt.savefig('a.png', bbox_inches='tight')
plt.show()
