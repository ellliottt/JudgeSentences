## Proposal

After Joseph's explanation, I have a much clear thought about our project. Sorry for the bad status before, because I totally confused about the defination of hashness. So the following are my thoughts, please help me complement it.(You can delete this part before sending the final one)

### Problem defination

So basically, we need to predict the jail's year according to the sentence data. So we can got : 

	yi=αJudgei+βX+α2type+ϵi
The problem is ϵi is not a random variable, so the prediction won't works well(We will take this as a baseline model). It is related to judge's bias(hashness,I take it as a bias). Judge_i variable can't represent this bias because there are many bias type so we need a feature representation. Take an example, judge A may be hasher on drug cases while more gentle on stolen cases.  
This bias can be reflected from judge's history opinions because his words will reflect his bias unintentionally.  
We can actually conclude this problem as a opinion mining problem(reference 11 joseph found is quiet a good ref.) It's very like predict the movie rate according to customer reviews. But there is some big difference such as Judges will try best to make the opinions fair and use more fact rather than adjective or emotion words. This makes this problem much harder.
Base on this assumption, this project is defined as following two parts:

**Part1**:
We need to build an opinion mining model which will embedding judge's opinions into feature vectors. Those vectors will improve the preformance of predicting the jail's year. After that ,we have a average for each cases, so if a result is significant higher above on average, it will be defined as hasher.  
**Part2**:
If we can predict the outcome more precisely, we can find which factors or which part of text contributes to hashness. Because the feature we make includes bias infomation. This is causal inference part. Once we got who is more hasher, we can train a model which will predict the hashness directly from opinions.  
We can focus on the part1 befor final exam.

## Methods and models 

For now, we don't consider the time effect and connection between opinions. It means we assume that judge's bias is a constant vector independent with time adn there is no inter-influence between judges. 

As we see, the most important part is how to transform the opinions into feature vectors which can improve the performance of model.

There are two ways to do this, and we can try them both.  

**Method1: Statistic based features**  
Those features are manully build according to the statistical information.
This includes:  
	
* N-grams features : unigram, bigram, trigrams.
* Pos analysis: we can extract the adjectives and other import words to decide the polarity degree.
* Negation words   
* Topic-related features
* Position infomation (pos +-1/ pos +-2)
* strong/weak subjectives
* Dependency parse

About those features, there is a good reference in artical 13 at the end of proposal.

**Word embedding method**:

The statistic model will works but has some issues like It costs too much time and spaces to compute the sparse matrix and human designed features are limited.
So we can try the embedding methods.  
For the exist methods we can use :  

* word2vec/ glove(much faster)
* doc2vec  

But we need to deal with a problem that ,what we need features of judge. So we need to embed those features above into one feature vector which represent each judge.  
Also, our model need to produce different feature vector by different case type(rob, drug..) even for same judge (judge has different bias on different type of case).  
**Attention model[14]** will help us to do that. This model(a kind of CNN/RNN model) will help us to decide which part of opinion is related to sentence result most.   
So we can actually build an end to end architecture which combine the feature extraction and prediction and train it together.  

## Schedule

* Build a baseline model only on sentence data 
* Next week: Accomplish statistic features and train again
* 4.3 - 4.9 Finish doc2vec and word2vec and embed those feature into Network.
* 4.10-4.16 Build attention model 
* Last : test, fine tune and finish report of part1
* After final exam, Part2



## some reference

1. [Encoding Sentences with Graph Convolutional Networks
for Semantic Role Labeling](https://arxiv.org/pdf/1703.04826.pdf)
2. [Sparse Named Entity Classification using Factorization Machines](https://arxiv.org/pdf/1703.04879.pdf)
3. [Deep Learning applied to NLP](https://arxiv.org/pdf/1703.03091.pdf)
4. [Thumbs up? Sentiment Classification using Machine Learning
Techniques](https://arxiv.org/pdf/cs/0205070.pdf)
5. [Opinion mining and sentiment analysis](http://www.cs.cornell.edu/home/llee/omsa/omsa.pdf)
6. [Deep Learning, NLP, and Representations](http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/)
7. [ICML 2015](http://icml.cc/2015/?page_id=97)
8. [multi-class-text-classification-cnn-rnn](https://github.com/jiegzhan/multi-class-text-classification-cnn-rnn)
9. [Name Entity recognition using LSTM](https://github.com/monikkinom/ner-lstm)
10. [Tensorflow realized attention machanism](http://weibo.com/1402400261/EBfUjCa2e?type=comment#_rnd1490385653151)
11. [Opinion mining and sentiment analysis](http://www.cs.cornell.edu/home/llee/omsa/omsa.pdf)
12. [Thumbs up? Sentiment Classification using Machine Learning
Techniques](https://arxiv.org/pdf/cs/0205070.pdf)
13. [Human bias detection](https://nlp.stanford.edu/pubs/neutrality.pdf)
14. [attention based](http://www.aclweb.org/anthology/D16-1058)
