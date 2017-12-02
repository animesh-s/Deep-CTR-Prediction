import torch
import torch.nn as nn
from torch.autograd import Variable

torch.manual_seed(1)

class  Xgb(nn.Module):
    def __init__(self, args):
        super(Xgb, self).__init__()
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
        
    def forward(self, line, dicts):
        x = self.feature(line, dicts)
        if self.args.static:
            x = Variable(x.data)
        x = x.data.numpy()[0]
        return x

    def feature(self, line, dicts):
        ip = line[dicts[1]['ip']].split('.')[:-1]
        userids = line[dicts[1]['userids']].split(',')
        userids = [dicts[3][9][x] for x in userids]
        url = 1 if line[dicts[1]['url']] != 'null' else 0
        aurl = 1 if line[dicts[1]['aurl']] != 'null' else 0
        adexchange = line[dicts[1]['adexchange']]
        adexchange = 0 if adexchange == 'null' else int(adexchange)
        ip1 = self.ip1_embed(variable(int(ip[0])))
        ip2 = self.ip2_embed(variable(int(ip[1])))
        ip3 = self.ip3_embed(variable(int(ip[2])))
        url = self.url_embed(variable(url))
        aurl = self.aurl_embed(variable(aurl))
        regionid = self.regionid_embed(variable(dicts[3][0][line[dicts[1]['regionid']]]))
        cityid = self.cityid_embed(variable(dicts[3][1][line[dicts[1]['cityid']]]))
        adexchange = self.adexchange_embed(variable(adexchange))
        adslotw = self.adslotw_embed(variable(dicts[3][2][line[dicts[1]['adslotw']]]))
        adsloth = self.adsloth_embed(variable(dicts[3][3][line[dicts[1]['adsloth']]]))
        adslotv = self.adslotv_embed(variable(dicts[3][4][line[dicts[1]['adslotv']]]))
        adslotfp = self.adslotfp_embed(variable(dicts[3][5][line[dicts[1]['adslotfp']]]))
        creativeid = self.creativeid_embed(variable(dicts[3][6][line[dicts[1]['creativeid']]]))
        bidprice = self.bidprice_embed(variable(dicts[3][7][line[dicts[1]['bidprice']]]))
        payprice = self.payprice_embed(variable(dicts[3][8][line[dicts[1]['payprice']]]))
        userids = self.userids_embed(Variable(torch.LongTensor(userids)))
        userids = torch.mean(userids, 0).view(1, -1)
        return torch.cat((ip1, ip2, ip3, url, aurl, regionid, cityid, 
                          adexchange, adslotw, adsloth, adslotv, adslotfp,
                          creativeid, bidprice, payprice, userids), 1)


def variable(x):
    return Variable(torch.LongTensor([x]))
