import random

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB


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
    features = features.toarray()



    v = TfidfVectorizer(analyzer='word', lowercase=True, stop_words='english')

    return data, labels, features


def regressor(x, y, _x, _y):
    regressor = LogisticRegression()

    regressor = regressor.fit(X=x, y=y)

    nbc = MultinomialNB()

    

    y_pred = regressor.predict(_x)
    res = accuracy_score(_y, y_pred)

    return y_pred, res


def normaliser(data):
    d = TfidfVectorizer()

