
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

# movie dataset

