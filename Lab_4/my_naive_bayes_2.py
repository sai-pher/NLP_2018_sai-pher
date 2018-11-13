#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pickle
import sys
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import nltk
from nltk.stem.lancaster import LancasterStemmer


# In[2]:


vectoriser_file = "vectoriser.pickle"
tfvectoriser_file = "tfvectoriser.pickle"
regressor_file = "regressor.pickle"
nbc_file = "nbc.pickle"


# In[3]:


def save(file, obj):
    with open(file, "wb") as f:
        pickle.dump(obj, f)

def load(file):
    with open(file, "rb") as f:
        model = pickle.load(f)
    return model

def check_file(*files):
    flag = True
    for file in files:
        try:
            f = open(file)
            f.close()
        except FileNotFoundError:
            flag = False
            pass
   
    return flag

def write(data, file="results.txt"):
    with open(file, "w") as f:
        for i in data:
            f.write("{}\n".format(i))


# In[4]:


def read_data(*files):
    data = []
    labels = []
    for file in files:
        temp = pd.read_csv(file, sep='\t', header=None)
        data.append(temp[0])
        labels.append(temp[1])
        
    data = pd.concat(data)
    labels = pd.concat(labels)
       
    vectorizer = CountVectorizer(analyzer='word', lowercase=False)
    features = vectorizer.fit_transform(data)
    
    save(vectoriser_file, vectorizer)
    
    features = features.toarray()
    
    return data, labels, features


def read_data_n(*files):
    data = []
    labels = []
    for file in files:
        temp = pd.read_csv(file, sep='\t', header=None)
        
        data.append(temp[0])
        labels.append(temp[1])
    
        
    data = pd.concat(data)
    labels = pd.concat(labels)
        
    vectorizer = TfidfVectorizer(use_idf=True, analyzer='word', strip_accents='ascii', lowercase=True, stop_words='english')
    features = vectorizer.fit_transform(data)
    
    save(tfvectoriser_file, vectorizer)
    
    features = features.toarray()
    
    return data, labels, features
    
    
def read_test_data(*files, state):
    data = []

    for file in files:
        temp = pd.read_csv(file, sep='\t', header=None)
        data.append(temp[0])
        
    data = pd.concat(data)
    
    if check_file(vectoriser_file, tfvectoriser_file):
        if state == 'n':
            vect = load(tfvectoriser_file)
        elif state == 'u':
            vect = load(vectoriser_file)
            
        features = vect.transform(data)
        
    return data, features
    


# In[5]:


def regressor(x, y, _x, _y):
    
    regressor = LogisticRegression()
    regressor = regressor.fit(X=x, y=y)
    y_pred = regressor.predict(_x)
    res = accuracy_score(_y, y_pred)
    
    save(regressor_file, regressor)
    
    return y_pred, res
    
def NB_Classifier(x, y, _x, _y):
    nbc = MultinomialNB()
    nbc = nbc.fit(X=x, y=y)
    y_pred = nbc.predict(_x)
    res = accuracy_score(_y, y_pred)
    
    save(nbc_file, nbc)
    
    return y_pred, res


# In[ ]:





# In[6]:


amazon = "data/amazon_cells_labelled.txt"
imdb = "data/imdb_labelled.txt"
yelp = "data/yelp_labelled.txt"

def train():
    data, lables, features = read_data(amazon, imdb, yelp)
    x_train, x_test, y_train, y_test = train_test_split(features, lables, test_size=0.2, random_state=0)
    
    pred, res = regressor(x_train, y_train, x_test, y_test)
    print("Unnormalised Linear Regression: {}%".format(round(res*100, 3)))
    pred, res = NB_Classifier(x_train, y_train, x_test, y_test)
    print("Unnormalised Naive Bayes: {}%".format(round(res*100, 3)))
    
    print("====================================")
    
    data, lables, features = read_data_n(amazon, imdb, yelp)
    x_train, x_test, y_train, y_test = train_test_split(features, lables, test_size=0.2, random_state=0)
    
    pred, res = regressor(x_train, y_train, x_test, y_test)
    print("Normalised Linear Regression: {}%".format(round(res*100, 3)))
    pred, res = NB_Classifier(x_train, y_train, x_test, y_test)
    print("Normalised Naive Bayes: {}%".format(round(res*100, 3)))
    
train()


# In[ ]:





# In[7]:


def main(argv):
    while not check_file(regressor_file, nbc_file, vectoriser_file):
        print("Some files are unavailable. Retraining.")
        train()
        print('Retraining successful!')

    regressor = load(regressor_file)
    nbc = load(nbc_file)

    if argv[1] == "nb" and argv[2] == "u":
        data, features = read_test_data(argv[3], "u")
        y = nbc.predict(features)
        write(y)
        
    if argv[1] == "nb" and argv[2] == "n":
        data, features = read_test_data(argv[3], "n")
        y = nbc.predict(features)
        write(y)
        
    if argv[1] == "lr" and argv[2] == "u":
        data, features = read_test_data(argv[3], "u")
        y = regressor.predict(features)
        write(y)
        
    if argv[1] == "lr" and argv[2] == "u":
        data, features = read_test_data(argv[3], "n")
        y = regressor.predict(features)
        write(y)
        
main(sys.argv)
        


# In[ ]:




