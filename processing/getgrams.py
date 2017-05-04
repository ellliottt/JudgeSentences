# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 05:43:00 2016

@author: elliott
"""

import os
from glob import glob
from nltk import bigrams as bigramize, word_tokenize, PorterStemmer
import pandas as pd
import re
from collections import Counter

os.chdir('/home/research/corpora/court-listener/data/district-courts/text')
folders = glob('*')

porter = PorterStemmer()
from nltk.corpus import stopwords
stoplist = set(porter.stem(w) for w in stopwords.words('english'))

word2id = pd.read_pickle('/home/research/projects/Ash_Chen_Naidu/data/word2id-2.pkl')

for folder in folders:        
    members = glob(folder + '/*txt')
    print(folder,len(members))
    if len(members) == 0:
        continue
    casefreqs = {}
    caselengths_all = {}
    caselengths_filtered = {}
    for fname in members:
        
        if not fname.endswith('txt'):
            continue

        cluster = fname.split('/')[-1][:-4]    
        text = open(fname).read()    
        normtext = re.sub('[^a-z0-9 ]','',text.lower())
        tokens = word_tokenize(normtext)        
        tokens = [porter.stem(w) for w in tokens if w.isalpha() and w not in stoplist]        
        bigrams = ['_'.join(b) for b in bigramize(tokens)]
        caselengths_all[cluster] = len(bigrams)
        tokenids = [word2id[b] for b in bigrams + tokens if b in word2id]
        #tokenids = [word2id[b] for b in bigrams + tokens if b in word2id]
        caselengths_filtered[cluster] = len(tokenids)
        casefreqs[cluster] = Counter(tokenids)
                
    pd.to_pickle(casefreqs,'/home/research/projects/Ash_Chen_Naidu/data/district-court-cases/grams/casefreqs-%s.pkl'%folder)
    pd.to_pickle(caselengths_all,'/home/research/projects/Ash_Chen_Naidu/data/district-court-cases/grams/caselengths_all-%s.pkl'%folder)
    pd.to_pickle(caselengths_filtered,'/home/research/projects/Ash_Chen_Naidu/data/district-court-cases/grams/caselengths_filtered-%s.pkl'%folder)