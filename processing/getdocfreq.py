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
from zipfile import ZipFile
from random import shuffle

os.chdir('/home/research/corpora/court-listener/data/district-courts/text')
folders = glob('*')

porter = PorterStemmer()
from nltk.corpus import stopwords
stoplist = set(porter.stem(w) for w in stopwords.words('english'))

for folder in folders:        
    members = glob(folder + '/*txt')
    print(folder,len(members))
    if len(members) == 0:
        continue
    threshold = len(members) / 200    
    docfreqs = Counter()        
    for fname in members:
        
        cluster = fname.split('/')[-1][:-4]    
        text = open(fname).read()    
        normtext = re.sub('[^a-z]',' ',text.lower())
        tokens = normtext.split()     
        tokens = [porter.stem(w) for w in tokens if len(w) >=3 and w not in stoplist]        
        bigramset = set(['_'.join(b) for b in bigramize(tokens)])
                      
        docfreqs.update(bigramset)
        
    for k,v in docfreqs.most_common():
        if v < threshold:
            del docfreqs[k]
            
    pd.to_pickle(docfreqs,'docfreqs-%s.pkl'%folder)
    
        