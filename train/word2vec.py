from gensim.models import word2vec
import pandas as pd

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


EMBEDDING_DIM = 300
texts = pd.read_pickle('datasets/lstmdata/textsfortoken.pkl')

class MySentences(object):
    def __init__(self, texts):
        self.texts = texts
 
    def __iter__(self):
        for text in self.texts:
            yield text.split(' ')

sentences = MySentences(texts) # a memory-friendly iterator
model = word2vec.Word2Vec(sentences, size = EMBEDDING_DIM)



model.save('datasets/word2vec_300.model')