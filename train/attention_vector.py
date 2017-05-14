
# coding: utf-8
from gensim import corpora, models, similarities
from gensim import corpora
from gensim.models import word2vec
import gensim
import pandas as pd
import numpy as np
import os
import os.path
from glob import glob
from collections import Counter
import matplotlib.pyplot as plt
import pickle

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras.models import Model

from keras import regularizers
from keras import backend as K
from keras.engine.topology import Layer, InputSpec

from keras import initializations
import argparse

import tensorflow as tf



ap = argparse.ArgumentParser()
ap.add_argument("-n", "--epochs", default = 100, type=int,
    help="num of epoch")
ap.add_argument("-batch", "--batchsize", default = 16,type=int,
    help="batch size")
ap.add_argument('-MAX_SENTS', "--MAX_SENTS", default = 50, type=int,
    help='MAX SENTENCE len')
ap.add_argument('-EMBEDDING_DIM', "--EMBEDDING_DIM", default = 100, type=int,
    help='EMBEDDING_DIM')
args = vars(ap.parse_args())




class AttLayer(Layer):
    def __init__(self, **kwargs):
        self.init = initializations.get('normal')
        #self.input_spec = [InputSpec(ndim=3)]
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape)==3
        self.W = self.init([input_shape[-1]])
        self.b = self.init([1])
        #self.W = tf.contrib.keras.initializers.RandomNormal()([input_shape[-1]])
        #self.b = tf.contrib.keras.initializers.RandomNormal()([1])
        #self.input_spec = [InputSpec(shape=input_shape)]
        self.trainable_weights = [self.W, self.b]
        super(AttLayer, self).build(input_shape)  # be sure you call this somewhere!

    def call(self, x, mask=None):
        eij = K.tanh(x * self.W + self.b)
        
        ai = K.exp(eij)
        weights = ai/tf.expand_dims(K.sum(ai, axis=1),1)
        
        weighted_input = x*weights
        return tf.reduce_sum(weighted_input ,axis=1)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[-1])

def vector_loss(y_true, y_pred):
    '''Just another crossentropy'''
    index = 1.0 - tf.cast(tf.equal(y_true, 0.) , tf.float32)
    num = tf.reduce_sum(index) * 1.0 + 0.00001
    return tf.reduce_sum((y_true * index - y_pred * index)**2) / num

# In[19]:

lstm_train = np.load("../datasets/lstmdata/lstm_train.npy")
lstm_test = np.load("../datasets/lstmdata/lstm_test.npy")
y_train = np.load("../datasets/lstmdata/y_train.npy")
y_test = np.load("../datasets/lstmdata/y_test.npy")

print('load datasets sucessfully')

# In[20]:

EMBEDDING_DIM = args['EMBEDDING_DIM']
MAX_SENTS = args['MAX_SENTS']
MAX_SENT_LEN = 30
MAX_NB_WORDS = 500000

# In[21]:

model = word2vec.Word2Vec.load('../datasets/word2vec.model')
filepath="../datasets/lstmdata/weights.best.hdf5"
if os.path.isfile(filepath):
    model.load_weights(filepath)
# In[ ]:
'''
texts = pd.read_pickle('../datasets/lstmdata/textsfortoken.pkl')
tokenizer = Tokenizer(nb_words=None)
print('load texts tokens sucessfully')
tokenizer.fit_on_texts(texts)
word_index = tokenizer.word_index
'''
tokenizer = pickle.load(open('../datasets/lstmdata/tokenizer.out', 'rb'))
word_index = tokenizer.word_index
# In[ ]:

embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    try:
        embedding_vector = model.wv[word]
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    except KeyError as err:
        embedding_matrix[i] = np.zeros(EMBEDDING_DIM)
        
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SENT_LEN,
                            trainable=True)


# In[14]:


sentence_input = Input(shape=(MAX_SENT_LEN,), dtype='int32')
embedded_sequences = embedding_layer(sentence_input)
l_lstm = Bidirectional(GRU(100, return_sequences=True))(embedded_sequences)
l_dense = TimeDistributed(Dense(200))(l_lstm)
l_att = AttLayer()(l_dense)
sentEncoder = Model(sentence_input, l_att)

review_input = Input(shape=(MAX_SENTS,MAX_SENT_LEN), dtype='int32')
review_encoder = TimeDistributed(sentEncoder)(review_input)
l_lstm_sent = Bidirectional(GRU(100, return_sequences=True))(review_encoder)
l_dense_sent = TimeDistributed(Dense(200))(l_lstm_sent)
l_att_sent = AttLayer()(l_dense_sent)
preds = Dense(1, activation=None)(l_att_sent)
model = Model(review_input, preds)

model.compile(loss= vector_loss,
              optimizer='adam',
              metrics=['mse','mae'])

checkpoint = ModelCheckpoint(filepath, monitor='val_mse', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

print("model fitting - Hierachical attention network")
history =model.fit(lstm_train, y_train, validation_data=(lstm_test, y_test),
        nb_epoch = args['epochs'], batch_size=args['batchsize'], callbacks=callbacks_list, verbose=1)

filep = open('../datasets/lstmdata/history.out', 'wb')
pickle.dump(history.history, filep)

model_json = model.to_json()
with open("../datasets/lstmdata/model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("../datasets/lstmdata/model.h5")




# In[ ]:



