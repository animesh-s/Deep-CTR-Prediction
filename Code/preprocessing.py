# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 21:01:07 2017

@author: Archit
"""

import bz2
import pickle

def read_data(filepath):
    data = []
    with bz2.BZ2File(filepath) as f:
        for line in f:
            line = line.split('\n')[0].split('\t')
            data.append(line)
    return data


def get_clkbidids():
    clkbidids = []
    for date in dates:
        filepath = '../Data/training3rd/clk.' + date + '.txt.bz2'
        with bz2.BZ2File(filepath) as f:
            for line in f:
                line = line.split('\n')[0].split('\t')
                clkbidids.append(line[0])
    return set(clkbidids)


def update_set_list(row):
    for i, key in enumerate(set_keys[:-1]):
        set_list[i].add(row[key2ind[key]])
    userid = set(row[key2ind['userids']].split(','))
    set_list[i + 1].update(userid)


def compute_set_list():
    for date in dates:
        filepath = '../Data/training3rd/imp.' + date + '.txt.bz2'
        with bz2.BZ2File(filepath) as f:
            for line in f:
                line = line.split('\n')[0].split('\t')
                update_set_list(line)


dates = ['201310' + str(i) for i in range(19, 28)]

keys = ['bidid', 'timestamp', 'logtype', 'ipinyouid', 'useragent', 'ip', \
        'regionid', 'cityid', 'adexchange', 'domain', 'url', 'aurl', \
        'adslotid', 'adslotw', 'adsloth', 'adslotv', 'adslotf', 'adslotfp', \
        'creativeid', 'bidprice', 'payprice', 'lpurl', 'advid', 'userids']

key2ind = dict((k, i) for i, k in enumerate(keys))

clkbidids = get_clkbidids()

regionids, cityids, adslotws, adsloths = set(), set(), set(), set()
adslotvs, adslotfps, creativeids, bidprices = set(), set(), set(), set()
payprices, userids = set(), set()
set_list = [regionids, cityids, adslotws, adsloths, adslotvs, adslotfps, \
            creativeids, bidprices, payprices, userids]
set_keys = ['regionid', 'cityid', 'adslotw', 'adsloth', 'adslotv', 'adslotfp', \
            'creativeid', 'bidprice', 'payprice', 'userids']

compute_set_list()

dict_list = [{val: idx for idx, val in enumerate(entry)} for entry in set_list]

alldicts_filepath = "../Processed Data/alldicts.pkl"

pickle.dump([clkbidids, key2ind, set_keys, dict_list], 
            open(alldicts_filepath, "wb"), pickle.HIGHEST_PROTOCOL)

#clkbidids, key2ind, set_keys, dict_list = pickle.load(open(alldicts_filepath, "rb"))
