# coding: utf-8

# In[1]:


import pickle
# Importing necessary libraries.
import sys

import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

topic = 'topicModel.pickle'
answer = 'answerModel.pickle'
t_vect = "topicVector.pickle"
a_vect = "answerVector.pickle"


# In[2]:


def preProcessing():
    # Opening .txt files
    q = open("Questions.txt", "r", encoding="utf8")
    a = open("Answers.txt", "r", encoding="utf8")
    t = open("Topics.txt", 'r', encoding="utf8")

    q_a = []
    a_a = []
    t_a = []

    with open('full.txt', 'w', encoding="utf8") as f:
        for line in q:
            q_a.append(str(line.rstrip('\n')))
        for line in a:
            a_a.append(str(line.rstrip('\n')))
        for line in t:
            t_a.append(str(line.rstrip('\n')))

        for i in range(len(q_a)):
            r = q_a[i] + "\t" + a_a[i] + "\t" + t_a[i] + "\n"
            f.write(r)

    q.close()
    a.close()
    t.close()

    dataList = pd.read_csv("full.txt", sep="\t", header=None)
    return dataList


# In[3]:


def vectorise(x, y, vect):
    normal = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents=ascii,
                             stop_words=set(stopwords.words('english')))
    cleanX = normal.fit_transform(x.astype(str))
    with open(vect, 'wb') as f:
        pickle.dump(normal, f)
    featureArr = cleanX.toarray()
    X_train, X_test, y_train, y_test = train_test_split(featureArr, y.astype(str), test_size=0.2, random_state=20)
    return X_train, X_test, y_train, y_test


# In[4]:


# This function creates a naive bayes model and trains on x_train and y_train.
# The trained model is stored in a pickle file.
def naiveBayes(x, y, model):
    nb = MultinomialNB()
    print('fitting.....')
    nb.fit(x, y)
    print('fitted!')
    with open(model, 'wb') as f:
        pickle.dump(nb, f)


# In[5]:


def train(model):
    data = preProcessing()
    q = data[0]
    a = data[1]
    t = data[2]
    if model == topic:
        values = vectorise(q, t, t_vect)
    elif model == answer:
        values = vectorise(q, a, a_vect)
    naiveBayes(values[0], values[2], model)


# In[6]:


def load(file):
    with open(file, 'rb') as f:
        trainedData = pickle.load(f)
    return trainedData


# In[7]:


def predict(test, mod):
    exist = False
    try:
        f = open(mod, 'rb')
        f.close()
        exist = True
    except FileNotFoundError:
        pass
    if exist is True:
        model = load(mod)
        pred = model.predict(test)
        pred = pred.tolist()
        for i in pred:
            print(str(i) + '\n')
    else:
        print('training {}.....'.format(mod))
        train(mod)
        model = load(mod)
        pred = model.predict(test)
        pred = pred.tolist()
        for i in pred:
            print(str(i) + '\n')
    return pred


# In[8]:


def toVector(file, vect):
    dataList = []

    df = pd.read_table(str(file), header=None)
    dataList.append(df)

    frame = pd.concat(dataList)

    vectorizer = load(vect)
    features = vectorizer.transform(frame[0])
    featureArr = features.toarray()
    return featureArr


# In[9]:


data = preProcessing()
q = data[0]
a = data[1]
t = data[2]
t_vector = vectorise(q, t, t_vect)
a_vector = vectorise(q, a, a_vect)


# In[10]:


def main(argv):
    if argv[1] == "qa":
        testData = toVector(argv[2], "answerVector.pickle")
        writeFile = open('qa_results.txt', 'w')
        exist = False
        try:
            f = open('answerModel.pickle', 'rb')
            f.close()
            exist = True
        except FileNotFoundError:
            pass
        if exist is True:
            model = load('answerModel.pickle')
            pred = model.predict(testData)
            pred = pred.tolist()
            for i in pred:
                writeFile.write(str(i) + '\n')
        else:
            train(answer)
            model = load('answerModel.pickle')
            pred = model.predict(testData)
            pred = pred.tolist()
            for i in pred:
                writeFile.write(str(i) + '\n')
    elif argv[1] == "topic":
        testData = toVector(argv[2], "topicVector.pickle")
        writeFile = open('topic_results.txt', 'w')
        exist = False
        try:
            f = open('topicModel.pickle', 'rb')
            f.close()
            exist = True
        except FileNotFoundError:
            pass
        if exist is True:
            model = load('topicModel.pickle')
            pred = model.predict(testData)
            pred = pred.tolist()
            for i in pred:
                writeFile.write(str(i) + '\n')
        else:
            train(topic)
            model = load('topicModel.pickle')
            pred = model.predict(testData)
            pred = pred.tolist()
            for i in pred:
                writeFile.write(str(i) + '\n')


main(sys.argv)
