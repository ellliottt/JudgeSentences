# -*- coding: utf-8 -*-

"""
This script includes all functions deal with text features.

"""

import pandas as pd
import numpy as np
import os
from glob import glob
from collections import Counter


os.chdir('/Users/xiaodiu/Documents/github/JudgeSentences/')

# Get bigram count num and existing features
# This script runs slow, keep the result.
case2judge = pd.read_pickle('rawdata/cluster2songername.pkl')
judge2case = dict(zip(case2judge.values(),case2judge.keys()))
files = glob('rawdata/case_level/*.pkl')
num = len(files)
judge2count_bigram = Counter()
i = 1
for file in files:
    #print('file :'+ file)
    case_freq = pd.read_pickle(file)
    for k,v in case_freq.items():
        k = int(k)
        if k in case2judge:
            judge = case2judge[k]
            #print('judge_name:' + judge)
            if(judge2count_bigram[judge] == 0):
                judge2count_bigram[judge] = Counter()
            judge2count_bigram[judge] += v
    if(i % 20 == 0 ):
        print('finished : %s' % (i))
    i += 1


# Get bigram count num and existing features
judge2exist_bigram = Counter()
for k, v in judge2count_bigram.items():
    judge2exist_bigram[k] = Counter(v.keys())

# save result
pd.to_pickle(judge2exist_bigram,
             'rawdata/judge2exist_bigram.pkl')
pd.to_pickle(judge2count_bigram,
             'rawdata/judge2count_bigram.pkl')