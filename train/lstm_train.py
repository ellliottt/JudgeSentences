
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


from keras import backend as K
from keras.engine.topology import Layer, InputSpec

from keras import initializations



# In[19]:

lstm_train = np.load("../datasets/lstmdata/lstm_train.npy")
lstm_test = np.load("../datasets/lstmdata/lstm_test.npy")
y_train = np.load("../datasets/lstmdata/y_train.npy")
y_test = np.load("../datasets/lstmdata/y_test.npy")

print('load datasets sucessfully')

# In[20]:

EMBEDDING_DIM = 300
MAX_SENTS = 500
MAX_SENT_LEN = 100
MAX_NB_WORDS = 500000

lstm_train = lstm_train[:,:MAX_SENTS,:]
lstm_test = lstm_test[:,:MAX_SENTS,:]

# In[21]:

model = word2vec.Word2Vec.load('../datasets/word2vec.model')
filepath="../datasets/lstmdata/weights.best.hdf5"
if os.path.isfile(filepath):
    model.load_weights(filepath)
# In[ ]:

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
l_lstm = Bidirectional(LSTM(EMBEDDING_DIM))(embedded_sequences)
#l_dense = TimeDistributed(Dense(200))(l_lstm)
#l_att = AttLayer()(l_dense)
sentEncoder = Model(sentence_input, l_lstm)

review_input = Input(shape=(MAX_SENTS,MAX_SENT_LEN), dtype='int32')
review_encoder = TimeDistributed(sentEncoder)(review_input)
l_lstm_sent = Bidirectional(LSTM(EMBEDDING_DIM))(review_encoder)
#l_dense_sent = TimeDistributed(Dense(200))(l_lstm_sent)
#l_att_sent = AttLayer()(l_dense_sent)
preds = Dense(1, activation=None)(l_lstm_sent)
model = Model(review_input, preds)

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['mse','mae'])

checkpoint = ModelCheckpoint(filepath, monitor='val_mse', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

print("model fitting - Hierachical attention network")
history =model.fit(lstm_train, y_train, validation_data=(lstm_test, y_test),
        nb_epoch = 200, batch_size=16, callbacks=callbacks_list, verbose=1)

model_json = model.to_json()
with open("../datasets/lstmdata/model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("../datasets/lstmdata/model.h5")

file_history = open('../datasets/lstmdata/history.out','wb')
pickle.dump(history, file_history)
# summarize history for accuracy
fig = plt.figure()
plt.plot(history.history['mean_squared_error'])
plt.plot(history.history['mean_absolute_error'])
plt.title('model error')
plt.ylabel('error')
plt.xlabel('epoch')
plt.legend(['mse', 'mae'], loc='upper left')
fig.savefig('../result/accuracy.png')
# summarize history for loss
fig = plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
fig.savefig('../result/loss.png')


# In[ ]:



