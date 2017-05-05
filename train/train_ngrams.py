
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

from gensim import corpora, models, similarities
from gensim import corpora
import gensim

from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model
from sklearn import cross_decomposition
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import hstack, vstack

import os, sys
os.chdir('/Users/xiaodiu/Documents/github/JudgeSentences/train/')


class textfeature():
	"""docstring for ClassName"""
	def __init__(self, top_k = 200 ,sparse = False):
		self.vectorizer = DictVectorizer(sparse=sparse)
		self.tfidf = TfidfTransformer()
		self.top_k = top_k
		self.f_reg = SelectKBest(f_regression, k = self.top_k)
		self.acc = 0
		self.index2judge_year = Counter()

	def load_data(self, path , format = 'csv', index_col = 0):
		if format == 'csv':
			print('load sucess from '+ path)
			return pd.read_csv(path, index_col = index_col)
		if format == 'pkl':
			print('load sucess from '+ path)
			return pd.read_pickle(path)
		

	def process_data(self, data, judge_year_index ,bow_feature, 
					y_label = 'demean_harshness',
					drops = ['judgeid', 'demean_harshness','sentyr'], 
					istrain = True):
		y_dict = {}
		ori = {}
		for judge in bow_feature.keys():
			d = {}
			for year in bow_feature[judge].keys():
				try:
					index = judge_year_index[judge][year]
				except TypeError:
					print(judge, year)
				logit = (data.sentyr == year) & (data.judgeid == judge)
				harshness = data[y_label][logit]
				if harshness.shape[0] == 0:
					continue
				y_dict[index] = harshness.values[0]
				ori[index] = list(data.drop(drops,axis = 1).loc[logit.index[logit == True][0]])
				d['judge'] = judge
				d['year'] = year
				self.index2judge_year[index] = d 

		if istrain:
			self.train_y_dict = y_dict
			self.train_ori = ori
		else:
			self.test_y_dict = y_dict
			self.test_ori = ori

	def get_train_test(self, bow_feature ):
		if self.train_ori is None:
			print('Process the data first')
			return 0
		train_X = {}
		test_X = {}
		for key in self.train_y_dict.keys():
			judge = self.index2judge_year[key]['judge']
			year = self.index2judge_year[key]['year']
			value = self.train_y_dict[key]
			train_X[key] = bow_feature[judge][year]

		for key in self.test_y_dict.keys():
			judge = self.index2judge_year[key]['judge']
			year = self.index2judge_year[key]['year']
			value = self.test_y_dict[key]
			test_X[key] = bow_feature[judge][year]

		self.train_X = train_X
		self.test_X = test_X

	def id2ngram(self, ngram_dict):
		self.id2ngram = {item[1]:item[0] for item in ngram_dict.items()}

	def get_vector(self):
		bow_train = self.vectorizer.fit_transform(list(self.train_X.values()))
		bow_test = self.vectorizer.transform(list(self.test_X.values()))
		return bow_train, bow_test

	def get_tfidf(self, bow_train, bow_test):
		X_train = self.tfidf.fit_transform(bow_train)
		X_test = self.tfidf.transform(bow_test)
		return X_train, X_test

	def selectKbest_subset(self, bow_train, bow_test):
		label = list(self.train_y_dict.values())
		best_train = self.f_reg.fit_transform(bow_train, label)
		best_test = self.f_reg.transform(bow_test)
		return best_train, best_test

	def combine_data(self, X_train, X_test):
		train_total = hstack([np.array(list(self.train_ori.values())), X_train])
		test_total = hstack([np.array(list(self.test_ori.values())), X_test])
		return train_total, test_total

	def get_k_features(k = 200):
		feature_names = self.vectorizer.get_feature_names()
		return [id2ngram[feature_names[i]] for i in f_reg.get_support(indices = True)]
		
	def model_pre(self, reg, X_train, X_test, scale = 1.0, dataset = 'Holger', model_name = 'regression'):
		#self.model = reg()
		y_train = np.array(list(self.train_y_dict.values())) * scale
		y_test = np.array(list(self.test_y_dict.values())) * scale
		reg.fit(X_train, y_train)
		y_pre = reg.predict(X_test)
		self.acc = np.mean((y_pre - y_test) ** 2)
		print(("Mean squared error for dataset %s on %s : %.4f" % (dataset, model_name, self.acc)))
		return y_pre

	def plot_scatter(self, y_pre, save_path = '../result/', scale = 1.0, name = 'bow_feature.png'):
		y_test = np.array(list(self.test_y_dict.values())) * scale
		fig = plt.figure()
		plt.scatter(y_test, y_pre)
		plt.xlabel('true value')
		plt.ylabel('prediction value')
		savename = save_path + name +'.png'
		fig.savefig(savename)

	def run_model(self, train_data, test_data, judge_year_index,
				  features, model_zoo, plot_name = 'bow_features', 
				  drops = ['judgeid', 'demean_harshness','sentyr'],
				  y_label = 'demean_harshness', 
				  dataset = 'Holger', model_name = 'OLS'):
		self.process_data(train_data, judge_year_index, features, 
						  istrain = True, drops = drops,
						  y_label = y_label)
		self.process_data(test_data, judge_year_index, features, 
						  istrain = False, drops = drops,
						  y_label = y_label)
		self.get_train_test(features)
		bow_train, bow_test = self.get_vector()
		X_train, X_test = self.get_tfidf(bow_train, bow_test)
		train_total, test_total = self.combine_data(X_train, X_test)


		## model test
		y_pre = self.model_pre(model_zoo[model_name], train_total, test_total, dataset = dataset, model_name = model_name)

		## plot
		self.plot_scatter(y_pre, name = plot_name+'_'+model_name+'_'+dataset)

if __name__ == '__main__':
	ngrams = textfeature()

	## load data
	train_data = ngrams.load_data('../holger_train_judgeyear.csv',index_col=0)
	test_data = ngrams.load_data('../holger_test_judgeyear.csv', index_col=0)
	judge_year_index = ngrams.load_data('../datasets/judge_year2index.pkl', format = 'pkl')
	ngram_dict = ngrams.load_data('../datasets/grams_dict2002-2016/grams_dict.pkl', format = 'pkl')

	bow_feature = ngrams.load_data('../datasets/grams_dict2002-2016/bow_features.pkl', format = 'pkl')
	bi_feature = ngrams.load_data('../datasets/grams_dict2002-2016/2grams_feature.pkl', format = 'pkl')
	tri_feature = ngrams.load_data('../datasets/grams_dict2002-2016/3grams_feature.pkl', format = 'pkl')
	for_feature = ngrams.load_data('../datasets/grams_dict2002-2016/4grams_feature.pkl', format = 'pkl')
	fiv_feature = ngrams.load_data('../datasets/grams_dict2002-2016/5grams_feature.pkl', format = 'pkl')

	model_zoo = Counter()
	model_zoo['OLS'] = linear_model.LinearRegression()
	model_zoo['PLS'] = cross_decomposition.PLSRegression(n = 200)
	model_zoo['RF']  = RandomForestRegressor()
	model_zoo['Elastic Net'] = linear_model.ElasticNet(alpha=0.1, l1_ratio=0.7)
	

	ngrams.run_model(train_data, test_data, judge_year_index, bow_feature)
















