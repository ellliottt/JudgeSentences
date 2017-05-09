from gensim.models import word2vec
import pandas as pd
from glob import glob
import logging
import numpy as np
import gensim
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

'''
EMBEDDING_DIM = 100
texts = pd.read_pickle('datasets/lstmdata/textsfortoken.pkl')

class MySentences(object):
	def __init__(self, texts):
		self.texts = texts
 
	def __iter__(self):
		for text in self.texts:
			yield text.split(' ')

sentences = MySentences(texts) # a memory-friendly iterator
model = word2vec.Word2Vec(sentences, size = EMBEDDING_DIM)



model.save('datasets/word2vec.model')

'''
class MyDoc(object):
	def __init__(self, path):
		self.folders = glob(path + '*')

 
	def __iter__(self):
		for fold in self.folders:
			members = glob(fold + '/*txt')
			for fname in members:
				if not fname.endswith('txt'):
					continue
				with open(fname) as f:
					for i, line in enumerate(f):
						yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [np.random.rand() - 0.5])


docs = MyDoc('rawdata/text/akd')
model = gensim.models.doc2vec.Doc2Vec(docs, size=100, min_count=2)

#model.build_vocab(train_corpus)

model.save('datasets/doc2vec.model')