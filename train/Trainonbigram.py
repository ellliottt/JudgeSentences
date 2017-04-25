from gensim import corpora, models, similarities
from gensim import corpora
import gensim
import pandas as pd
import numpy as np
import os
from glob import glob
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
import sys
from sklearn import linear_model
sys.path.append(os.getcwd())



judge2count_bigram = pd.read_pickle('../rawdata/judge2count_bigram.pkl')

word2id = pd.read_pickle('../rawdata/word2id-2.pkl')

judge2count_bigram['dict'] = Counter(word2id.values())


v = DictVectorizer(sparse=False)
judge2count_bigram_matrix = \
    v.fit_transform(list(judge2count_bigram.values()))


data = pd.read_csv('../Holger_train.csv',decimal=',',index_col=0)
test = pd.read_csv('../Holger_test.csv',decimal=',',index_col=0)



drops = ['judgeid', 'demean_logsenttot','cr22','malejudge']
for i in range(10):
    drops.append('year' + str(i+1))
for i in range(8):
    drops.append('race' + str(i+1))


## base line
regr = linear_model.LinearRegression(fit_intercept = False)
regr.fit(X_train, Y_train)
print("Mean squared error，base line: %.2f"
      % np.mean((regr.predict(X_test) - Y_test) ** 2))



predictions = []
improvements = []
i = 0
for k, v in word2id.items():
    feature = pd.DataFrame(judge2count_bigram_matrix[:, v-1], columns=[k])
    feature['judgeid'] = np.array(list(judge2count_bigram.keys()))
    Train_data = pd.merge(data, feature, how='inner', on='judgeid')
    X_train = Train_data.drop(drops, axis = 1)
    X_train['intercept'] = 1
    Y_train = np.array(Train_data['demean_logsenttot'], dtype=np.float32)
    
    Test_data = pd.merge(test, feature, how='inner', on='judgeid')
    X_test = Test_data.drop(drops, axis = 1)
    X_test['intercept'] = 1
    Y_test = np.array(Test_data['demean_logsenttot'] ,dtype=np.float32)
    
    
    regr.fit(X_train, Y_train)
    accuracy = np.mean((regr.predict(X_test) - Y_test) ** 2)
    predictions.append(accuracy)
    improvements.append((accuracy - 1.12)/ accuracy)

    i += 1
    if(i % 1000 ==0 ):
        print(i)

words = np.array(list(word2id.keys()))
predictions = np.array(predictions)
improvements = np.array(improvements)

predict_df = pd.DataFrame({"words" : words[:len(predictions)], 
                           "predictions" : predictions, 
                           "improvements" : improvements
                          })

predict_df = predict_df.sort_values(['improvements'],ascending=False)
predict_df.to_csv('../result/bigram_prediction_on_Hogler_data.csv')





