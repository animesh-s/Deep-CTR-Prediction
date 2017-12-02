# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 20:25:05 2017

@author: Archit
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

torch.manual_seed(1)


class LR(nn.Module):
    def __init__(self, args):
        super(LR, self).__init__()
        self.args = args
        factor = args.factor
        self.ip1_embed = nn.Embedding(256, factor * 8)
        self.ip2_embed = nn.Embedding(256, factor * 8)
        self.ip3_embed = nn.Embedding(256, factor * 8)
        self.regionid_embed = nn.Embedding(35, factor * 6)
        self.cityid_embed = nn.Embedding(370, factor * 9)
        self.adexchange_embed = nn.Embedding(9, factor * 4)
        self.url_embed = nn.Embedding(2, factor * 1)
        self.aurl_embed = nn.Embedding(2, factor * 1)
        self.adslotw_embed = nn.Embedding(21, factor * 5)
        self.adsloth_embed = nn.Embedding(14, factor * 4)
        self.adslotv_embed = nn.Embedding(7, factor * 3)
        self.adslotfp_embed = nn.Embedding(275, factor * 9)
        self.creativeid_embed = nn.Embedding(57, factor * 6)
        self.bidprice_embed = nn.Embedding(2, factor * 1)
        self.payprice_embed = nn.Embedding(295, factor * 9)
        self.userids_embed = nn.Embedding(69, factor * 7)
        self.linear = nn.Linear(factor * 89, 2)
        
    def forward(self, line, dicts, infer = False):
        x = feature(line, dicts, self)
        x = torch.cat(x, 1)
        if self.args.static:
            x = Variable(x.data)
        return self.linear(x)


class CNN(nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()
        D = args.embed_dim
        Co = args.kernel_num
        Ks = args.kernel_sizes
        self.args = args
        self.ip1_embed = nn.Embedding(256, D)
        self.ip2_embed = nn.Embedding(256, D)
        self.ip3_embed = nn.Embedding(256, D)
        self.regionid_embed = nn.Embedding(35, D)
        self.cityid_embed = nn.Embedding(370, D)
        self.adexchange_embed = nn.Embedding(9, D)
        self.url_embed = nn.Embedding(2, D)
        self.aurl_embed = nn.Embedding(2, D)
        self.adslotw_embed = nn.Embedding(21, D)
        self.adsloth_embed = nn.Embedding(14, D)
        self.adslotv_embed = nn.Embedding(7, D)
        self.adslotfp_embed = nn.Embedding(275, D)
        self.creativeid_embed = nn.Embedding(57, D)
        self.bidprice_embed = nn.Embedding(2, D)
        self.payprice_embed = nn.Embedding(295, D)
        self.userids_embed = nn.Embedding(69, D)
        self.Ci = 1
        self.convs = nn.ModuleList([nn.Conv2d(self.Ci, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(args.dropout)
        self.linear = nn.Linear(Co * len(Ks), 2)

    def forward(self, line, dicts, infer = False):
        x = self.feature_enc(line, dicts)
        if infer:
            x = x * self.args.dropout
        else:
            x = self.dropout(x)
        return self.linear(x)

    def feature_maps(self, x, convs):
        num_words = x.size(2)
        temp = Variable(torch.zeros((self.Ci, self.args.kernel_num, 1)))
        temp = temp.cuda() if self.args.cuda else temp
        y = []
        for conv, k in zip(convs, self.args.kernel_sizes):
            if k > num_words:
                y.append(temp)
            else:
                y.append(F.relu(conv(x)).squeeze(3))
        return y

    def feature_enc(self, line, dicts):
        x = feature(line, dicts, self)
        x = torch.cat(x, 0)
        if self.args.static:
            x = Variable(x.data)
        x = x.unsqueeze(0).unsqueeze(0)
        x = self.feature_maps(x, self.convs)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        return x


def feature(line, dicts, model):
    ip = line[dicts[1]['ip']].split('.')[:-1]
    userids = line[dicts[1]['userids']].split(',')
    userids = [dicts[3][9][x] for x in userids]
    url = 1 if line[dicts[1]['url']] != 'null' else 0
    aurl = 1 if line[dicts[1]['aurl']] != 'null' else 0
    adexchange = line[dicts[1]['adexchange']]
    adexchange = 0 if adexchange == 'null' else int(adexchange)
    ip1 = model.ip1_embed(
            variable(int(ip[0])))
    ip2 = model.ip2_embed(
            variable(int(ip[1])))
    ip3 = model.ip3_embed(
            variable(int(ip[2])))
    url = model.url_embed(
            variable(url))
    aurl = model.aurl_embed(
            variable(aurl))
    regionid = model.regionid_embed(
            variable(dicts[3][0][line[dicts[1]['regionid']]]))
    cityid = model.cityid_embed(
            variable(dicts[3][1][line[dicts[1]['cityid']]]))
    adexchange = model.adexchange_embed(
            variable(adexchange))
    adslotw = model.adslotw_embed(variable(
            dicts[3][2][line[dicts[1]['adslotw']]]))
    adsloth = model.adsloth_embed(variable(
            dicts[3][3][line[dicts[1]['adsloth']]]))
    adslotv = model.adslotv_embed(variable(
            dicts[3][4][line[dicts[1]['adslotv']]]))
    adslotfp = model.adslotfp_embed(
            variable(dicts[3][5][line[dicts[1]['adslotfp']]]))
    creativeid = model.creativeid_embed(
            variable(dicts[3][6][line[dicts[1]['creativeid']]]))
    bidprice = model.bidprice_embed(
            variable(dicts[3][7][line[dicts[1]['bidprice']]]))
    payprice = model.payprice_embed(
            variable(dicts[3][8][line[dicts[1]['payprice']]]))
    userids = model.userids_embed(
            Variable(torch.LongTensor(userids)))
    userids = torch.mean(userids, 0).view(1, -1)
    return (ip1, ip2, ip3, url, aurl, regionid, cityid, adexchange, adslotw,
            adsloth, adslotv, adslotfp, creativeid, bidprice, payprice, userids)


def variable(x):
    return Variable(torch.LongTensor([x]))
