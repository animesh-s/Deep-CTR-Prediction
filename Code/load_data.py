# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 10:02:50 2017

@author: Archit
"""

import bz2
import pickle
#import cPickle as pickle
import csv
import string
import time
from collections import Counter
import numpy as np

def read_data(filepath):
    #filepath = '../Data/training3rd/bid.20131019.txt.bz2'
    data = []
    with bz2.BZ2File(filepath) as f:
        for line in f:
            line = line.split('\n')[0].split('\t')
            data.append(line)
    return data

def read_data2(filepath):
    #filepath = '../Data/training3rd/bid.20131019.txt/bid.20131019.txt'
    with open(filepath) as f:
        reader = csv.reader(f, delimiter="\t")
        data = list(reader)
    return data

def read_data3(filepath):
    #filepath = '../Data/training3rd/bid.20131019.txt.bz2'
    zipfile = bz2.BZ2File(filepath) # open the file
    reader = zipfile.read() # get the decompressed data
    newfilepath = filepath[:-4] # assuming the filepath ends with .bz2
    open(newfilepath, 'wb').write(reader) # write a uncompressed file
    with open(newfilepath) as f:
        reader = csv.reader(f, delimiter="\t")
        data = list(reader)
    return data

def filename(typ, date):
    return '../Data/training3rd/' + typ + '.' + date + '.txt.bz2'

def get_raw_data(date):
    data = {}
    for i, typ in enumerate(types):
        data[typ] = read_data(filename(typ, date))
    return data

def get_data_verified(date):
    col_size = [24, 24]     # [21, 24, 24, 24]
    log_type = {'imp': '1', 'clk': '2', 'conv': '3'}
    data = {}
    for i, typ in enumerate(types):
        data[typ] = read_data(filename(typ, date))
        for j, row in enumerate(data[typ]):
            if len(row) != col_size[i]:
                print(typ, j)
            if i > 0 and row[2] != log_type[typ]:
                print(typ, row[2], log_type[typ])
    return data

def merge_str(str1, str2):
    if str1 == 'null':
        return str2
    elif str2 == 'null':
        return str1
    else:
        return str1 + ',' + str2

def process_data(data):
    clkbidids = set([row[key2ind['bidid']] for row in data['clk']])
    seenbidids = set()
    processed_data = {}
    for row in data['imp']:
        if row[key2ind['bidid']] not in seenbidids:
            seenbidids.add(row[key2ind['bidid']])
            if row[key2ind['bidid']] in clkbidids:
                row[key2ind['logtype']] = '2'
                #label2bid[1].append(row[key2ind['bidid']])
            else:
                pass
                #label2bid[0].append(row[key2ind['bidid']])
            processed_data[row[key2ind['bidid']]] = row
            #update_sets(row)
        else:
            str1 = processed_data[row[key2ind['bidid']]][key2ind['userids']]
            str2 = row[key2ind['userids']]
            processed_data[row[key2ind['bidid']]][key2ind['userids']] = merge_str(str1, str2)
    #pickle.dump(processed_data, open("bid2data.pkl", "ab"), pickle.HIGHEST_PROTOCOL)
    return processed_data.values()

def get_data():
    for date in dates:
        print(date)
        data = get_raw_data(date)
        process_data(data)
        
def get_data_by_date(date):
    data = get_raw_data(date)
    return process_data(data)

def update_sets(row):
    regionids.add(row[key2ind['regionid']])
    cityids.add(row[key2ind['cityid']])
    #adslotids.add(row[key2ind['adslotid']]) #53571
    adslotws.add(row[key2ind['adslotw']]) #21
    adsloths.add(row[key2ind['adsloth']]) #14
    adslotvs.add(row[key2ind['adslotv']]) #7
    adslotfps.add(row[key2ind['adslotfp']]) #275
    creativeids.add(row[key2ind['creativeid']]) #57
    bidprices.add(row[key2ind['bidprice']]) #2
    payprices.add(row[key2ind['payprice']]) #295
    userid = set(row[key2ind['userids']].split(','))
    userids.update(userid) #69
    """
    useragent = row[key2ind['useragent']]
    nodigit = ''.join([i for i in useragent if not i.isdigit()])
    useragent = nodigit.lower().translate(None, string.punctuation).split()
    useragents.update(set(useragent))
    """

dates = ['201310' + str(i) for i in range(19, 28)]
types = ['imp', 'clk']
keys = ['bidid', 'timestamp', 'logtype', 'ipinyouid', 'useragent', 'ip', \
        'regionid', 'cityid', 'adexchange', 'domain', 'url', 'aurl', \
        'adslotid', 'adslotw', 'adsloth', 'adslotv', 'adslotf', 'adslotfp', \
        'creativeid', 'bidprice', 'payprice', 'lpurl', 'advid', 'userids']
#ind2key = dict((i,k) for i,k in enumerate(keys))
key2ind = dict((k, i) for i, k in enumerate(keys))
regionids, cityids, adslotws, adsloths, adslotvs, adslotfps, \
creativeids, bidprices, payprices, userids = pickle.load(open("alldicts.pkl", "rb"))[1:]

for date in dates[:2]:
    print(date)
    data = get_data_by_date(date)


"""
label2bid = {0: [], 1: []}
regionids, cityids, adslotws, adsloths, adslotvs, adslotfps, creativeids, \
bidprices, payprices, userids = set(), set(), set(), set(), set(), set(), \
set(), set(), set(), set()

get_data()

regionids = dict((v, k) for k, v in enumerate(regionids))
cityids = dict((v, k) for k, v in enumerate(cityids))
adslotws = dict((v, k) for k, v in enumerate(adslotws))
adsloths = dict((v, k) for k, v in enumerate(adsloths))
adslotvs = dict((v, k) for k, v in enumerate(adslotvs))
adslotfps = dict((v, k) for k, v in enumerate(adslotfps))
creativeids = dict((v, k) for k, v in enumerate(creativeids))
bidprices = dict((v, k) for k, v in enumerate(bidprices))
payprices = dict((v, k) for k, v in enumerate(payprices))
userids = dict((v, k) for k, v in enumerate(userids))

pickle.dump([label2bid, regionids, cityids, adslotws, adsloths, adslotvs, \
             adslotfps, creativeids, bidprices, payprices, userids], \
    open("alldicts.pkl", "wb"), pickle.HIGHEST_PROTOCOL)

label2bid, regionids, cityids, adslotws, adsloths, adslotvs, adslotfps, \
creativeids, bidprices, payprices, userids = pickle.load(open("alldicts.pkl", "rb"))
"""

"""
bid_keys = ['bidid', 'timestamp', 'ipinyouid', 'useragent', 'ip', 'regionid', \
            'cityid', 'adexchange', 'domain', 'url', 'aurl', 'adslotid', \
            'adslotw', 'adsloth', 'adslotv', 'adslotf', 'adslotfp', \
            'creativeid', 'bidprice', 'advid', 'userids']
ind2bid = dict((i,k) for i,k in enumerate(bid_keys))
bid2ind = dict((k,i) for i,k in enumerate(bid_keys))
"""

"""
# Verification
bids = ['4d788b931f1864e83cf53fcb8f5ba72e','a5e66a096b976beae6b5b71e5590eb58']
for bid in bids:
    print(bid)
    for row in data:
        if row[0] == bid:
            print(row[key2ind['logtype']], row[key2ind['userids']])
"""

"""
count = 0
for row in data:
    if row[key2ind['logtype']] == '2':
        count += 1
print(count)
# 2700 clicks out of 3147801 impressions
"""

"""
a = [row[key2ind['bidid']] for row in data]
#c = [row.split('.')[3] for row in a]
b = list(set(a))
print('total', len(a), 'unique', len(b))

columns = [[]]*24
for row in data:
    for ind in range(24):
        columns[ind].append(row[ind])
counts = [[]]*24
for ind in range(2):
    counts[ind] = Counter(columns[ind])
for count in counts:
    counts[0].most_common(4)

bidid = count.most_common(5)[3][0]
datam2 = [row for row in data[typ] if row[dicts[types.index(typ)]['bidid']] == bidid]
d = zip(datam2[0], datam2[1], datam2[2])#, datam2[3]) #, datam2[4], datam2[5], datam2[6])
for i, x in enumerate(d):
    if len(set(x)) > 1:
        print(i, len(set(x)))

datam2 = [row for row in data[typ] if row[dicts[types.index(typ)]['adexchange']] == '1']

impbidid = [row[dicts[types.index(typ)]['bidid']] for row in data['imp']]
impbidid = list(set(impbidid))
convbidid = [row[dicts[types.index('conv')]['bidid']] for row in data['conv']]
bid = list(set(impbidid) - set(convbidid))

for bid in convbidid:
    print(bid)
    datam2 = [row for row in data['imp'] if row[dicts[types.index('imp')]['bidid']] == bid]
    datam3 = [row for row in data['conv'] if row[dicts[types.index('conv')]['bidid']] == bid]
    d = zip(datam2[0], datam3[0])#, datam2[3]) #, datam2[4], datam2[5], datam2[6])
    for i, x in enumerate(d):
        if len(set(x)) > 1:
            print(i, len(set(x)), x)

"""


#Observations
"""
bidid:
('bid', 'total', 352766, 'unique', 352766)
('imp', 'total', 228133, 'unique', 227827)
('clk', 'total', 83, 'unique', 83)
('conv', 'total', 16, 'unique', 16)
('bid', 'total', 326830, 'unique', 326830)
('imp', 'total', 214295, 'unique', 214022)
('clk', 'total', 65, 'unique', 65)
('conv', 'total', 22, 'unique', 22)
duplicates only in imp
duplicate bidids differ by timestamps always, ipinyouid(user cookie) in some cases
i.e., mutiple impressions with same bidid are created at different timestamps
with same ad opportunity

all bidids in imp log are present in bid log

adexchange:
unique 3: ['1', '3', '2']

adslotv:
['FourthView', 'Na', 'FirstView', 'FifthView', 'OtherView', 'ThirdView', 'SecondView']

adslotf:
['Na', 'Fixed', 'Pop']

advid:
single value   2259

userids:
null in bid, clk, conv log
present in imp log



todo:
Do everything for Alibaba ad exchange no. 1
1. create dict for adslotid(~15k), adslotw, adsloth, adslotv, \
, adslotfp, creativeid, bidprice(2), payprice(295), advid(4 in paper)  (add UNK at the end)
2. split ip address into first 3 parts and create dict (39, 222, 256, 1) (make it fixed size of 256)
3. for regionid, cityid, userids, read textfile and create dict; do averaging for userids
4. no idea about domain (~8k), url (~119k),
5. remove timestamp, ipinyouid, aurl(null), lpurl(null), adslotf('Na')
6. for useragent (~34k), split each and do average embedding
7. may add feature if url, aurl is present or not (0,1)
8. adslotw, adsloth can be arrays also (how to handle this)
"""
