# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 00:13:22 2017

@author: Archit
"""

#! /usr/bin/env python
import os
import argparse
import datetime
import torch
import train

parser = argparse.ArgumentParser(description='CTR Predictor')
# learning
parser.add_argument('-modeltype', type=str, default='CNN', help='model type (LR or CNN) [default: LR]')
parser.add_argument('-threshold', type=float, default=0.10, help='probability threshold [default: 0.1]')
#parser.add_argument('-lr', type=float, default=0.00859, help='learning rate to use for training')
#parser.add_argument('-weight-decay', type=float, default=0.00019, help='weight decay to use for training')
parser.add_argument('-optimizer', type=str, default='SGD', help='loss function optimizer (SGD or Adam) [default: SGD]')
parser.add_argument('-num-models', type=int, default=50, help='number of models for cross validation [default: 50]')
parser.add_argument('-epochs', type=int, default=5, help='number of epochs for train [default: 5]')
parser.add_argument('-iterations', type=int, default=None, help='number of iterations for train [default: None]')
parser.add_argument('-batch-size', type=int, default=32, help='number of examples in a batch [default:32]')
parser.add_argument('-imbalance-factor', type=int, default=9, help='class imbalance factor for training [default: 9]')
parser.add_argument('-log-interval',  type=int, default=6480,   help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-plot-interval',  type=int, default=300000,   help='how many steps to wait before plotting training status [default: 1]')
parser.add_argument('-save-interval', type=int, default=300000, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default='../Snapshots', help='where to save the snapshots')
parser.add_argument('-plot-dir', type=str, default='../Plots', help='where to save the plots')
parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
# LR model parameters
#parser.add_argument('-factor', type=int, default=100, help='factor for feature embeddings')
# CNN model parameters
parser.add_argument('-kernel-sizes', type=str, default='2,3,4,5', help='comma-separated kernel size to use for convolution')
#parser.add_argument('-dropout', type=float, default=0.1, help='the probability for dropout [default: 0.1]')
#parser.add_argument('-embed-dim', type=int, default=128, help='number of embedding dimension [default: 128]')
#parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
# device
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu' )
# option
args = parser.parse_args()


# update args and print
args.cuda = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda
args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
if args.iterations is None:
    args.iterations = args.epochs * (args.imbalance_factor + 1) * 2160
args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
args.plot_dir = os.path.join(args.plot_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir)
args.filepath = os.path.join(args.save_dir, 
                             args.modeltype + '_' + str(args.static) + '.txt')


print("\nParameters:")    
f = open(args.filepath, 'a')
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))
    f.write('%s = %s\n' %(attr.upper(), value))
f.close()


train.cross_validation(args)
