
# dependencies

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
import re
import os
from string import punctuation 
from textblob import Word 
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize

# add'nl databases

nltk.download('punkt') 

# tokenization
# divides a text into a list of sentences
# by using an unsupervised algorithm to build a model for abbreviation
# words, collocations, and words that start sentences.  It must be
# trained on a large collection of plaintext in the target language
# before it can be used.

nltk.download('stopwords') # stopwords
nltk.download('wordnet') # lexical database

# movie dataframes train, test and validation

# Must ensure path 
train = pd.read_csv('./IMDB/Train.csv')
test = pd.read_csv('./IMDB/Test.csv')
valid = pd.read_csv('./IMDB/Valid.csv')

# each df has three columns: index, text, and label.

# this function applies several lambda functions to movie dfs

def transformations(dataframe):
    #HTML Tags removal
    dataframe['text'] = dataframe['text'].apply(lambda words: re.sub('<.*?>','',words)) 
    #Word Tokenization
    dataframe['text'] = dataframe['text'].apply(word_tokenize)
    #Lower case conversion
    dataframe['text'] = dataframe['text'].apply(lambda words: [x.lower() for x in words])
    #Punctuation removal
    dataframe['text'] = dataframe['text'].apply(lambda words: [x for x in words if not x in punctuation])
    #Number removal
    dataframe['text'] = dataframe['text'].apply(lambda words: [x for x in words if not x.isdigit()])
    #Stopword removal
    dataframe['text'] = dataframe['text'].apply(lambda words: [x for x in words if x not in stopwords.words('english')])
    #Frequent word removal
    temp = dataframe['text'].apply(lambda words: " ".join(words))
    freq = pd.Series(temp).value_counts()[:10]
    dataframe['text'] = dataframe['text'].apply(lambda words: [x for x in words if x not in freq.keys()])
    #Lemmatization
    dataframe['text'] = dataframe['text'].apply(lambda words: " ".join([Word(x).lemmatize() for x in words]))
    return dataframe

#Applying Transformations
train = transformations(train)
valid = transformations(valid)
test = transformations(test)

# Put the text column into X and the label column into Y

X_train = train.text
Y_train = train.label
X_valid = valid.text
Y_valid = valid.label
X_test = test.text
Y_test = test.label

# save the model to disk
pickle.dump(classifier, open('LRClassifier.pkl', 'wb'))
# load the model from disk
loaded_model = pickle.load(open('LRClassifier.pkl', 'rb'))
result = loaded_model.score(X_test, y_test)
print(result)