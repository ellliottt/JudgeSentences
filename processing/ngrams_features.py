import os
from glob import glob
from nltk import bigrams as bigramize, word_tokenize, PorterStemmer, trigrams,ngrams
import pandas as pd
import re
from collections import Counter
from zipfile import ZipFile
from random import shuffle
from nltk.corpus import stopwords


def get_features(n = 2, year_range = (1978,2012)):
	bow_features = Counter()

	judges = judge2year_case.keys()
	for judge in judges:
		judge_grams = Counter()
		opinions = judge2year_case[judge]
		for year in opinions.keys():
			if year < year_range[0] or year > year_range[1]:
				continue
			opilist = opinions[year]
			year_grams = Counter()
			for opi in opilist:    
				fold = opinion2floder[str(opi)]
				filepath = 'rawdata/text/' + fold + '/' + str(opi) + '.txt'
				#print(filepath)
				text = open(filepath).read()    
				normtext = re.sub('[^a-z0-9 ]',' ',text.lower())
				tokens = normtext.split() 
				tokens = [porter.stem(w) for w in tokens if len(w) >=3 and w not in stoplist]  

				if n == 1:
					tokens = [grams_dict[w] for w in tokens if grams_dict.get(w) is not None]
					year_grams.update(tokens)
					continue

				if n == 2:
					gramsets = ['_'.join(b) for b in bigramize(tokens)]
					gramsets = [grams_dict[w] for w in gramsets if grams_dict.get(w) is not None]
					year_grams.update(gramsets)
				if n == 3:
					gramsets = ['_'.join(b) for b in trigrams(tokens)]
					gramsets = [grams_dict[w] for w in gramsets if grams_dict.get(w) is not None]
					year_grams.update(gramsets)
				if n > 3:
					gramsets = ['_'.join(b) for b in ngrams(tokens, n)]
					gramsets = [grams_dict[w] for w in gramsets if grams_dict.get(w) is not None]
					year_grams.update(gramsets)
			print(len(year_grams))
			judge_grams[year] = year_grams
		bow_features[judge] = judge_grams
	return bow_features

if __name__ == '__main__':
	folders = glob('rawdata/text/*')
	grams_dict = pd.read_pickle('datasets/grams_dict1978-2012/grams_dict.pkl')
	judge2year_case = pd.read_pickle('datasets/judge2year_case.pkl')
	opinion2floder = pd.read_pickle('datasets/opinion2floder.pkl')
	judges = judge2year_case.keys()
	print(len(folders))
	porter = PorterStemmer()
	stoplist = set(porter.stem(w) for w in stopwords.words('english'))

	for i in range(1,3):
		bow_features = get_features(n = i,year_range = (1978,2012))
		pd.to_pickle(bow_features, 'datasets/grams_dict1978-2012/%sgrams_feature.pkl'%str(i))


