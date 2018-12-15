#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pickle
import sys

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# In[ ]:


vectoriser_file = "vectoriser.pickle"
tfvectoriser_file = "tfvectoriser.pickle"
question_vectoriser_file = "qvtfvectoriser.pickle"
answer_vectoriser_file = "avtfvectoriser.pickle"
topic_vectoriser_file = "tvtfvectoriser.pickle"
regressor_file = "regressor.pickle"
regressor_n_file = "regressor_n.pickle"
nbc_file = "nbc.pickle"
nbc_n_file = "nbc_n.pickle"
model_path = "models/"
res_path = "results/"
data_path = "data/"


# In[ ]:


def save(file, obj):
    """
    Function to save objects to pickle files.

    :param file: File path to save to.
    :param obj: Object to be saved.
    """
    with open(file, "wb") as f:
        pickle.dump(obj, f)


def load(file):
    """
    Function to load objects from pickle files.

    :param file: File path to load from.
    :return: Object to load.
    """
    with open(file, "rb") as f:
        model = pickle.load(f)
    return model


def check_file(*files):
    """
    Function to check if a given list of files exist.

    :param files: Files to be checked.
    :return: Boolean True of False.
    """
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
    """
    Function to write data to a given text file.

    :param data: Data to be writen to a file.
    :param file: File path to be written to.
    """
    with open(file, "w") as f:
        for i in data:
            f.write("{}\n".format(i))


# In[ ]:

def data_process(questions, aanswers, topics):
    full = data_path + "full.txt"
    q_a = []
    a_a = []
    t_a = []

    with open(questions, "r") as q:
        for line in q:
            q_a.append(pd.read_csv(line, "\t", header=None))

    with open(aanswers, "r") as a:
        for line in a:
            a_a.append(pd.read_csv(line, "\t", header=None))

    with open(topics, "r") as t:
        for line in t:
            t_a.append(pd.read_csv(line, "\t", header=None))

    with open(full, "w") as f:
        for i in range(len(q_a)):
            l = "{} \t {} \t {} \n".format(q_a[i], a_a[i], t_a[i])
            f.write(l)


def read_data(*files):
    """
    Function to read and vectorise raw data from a list of given files.

    :param files: Files to be read from.
    :return: a tuple containing the raw data, its labels and the feature vectors.
    """
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

    save(model_path + vectoriser_file, vectorizer)

    features = features.toarray()

    return data, labels, features


def read_data_n(*files):
    """
    Function to read and vectorise raw data from a list of given files.
    This function utilises TF-IDF text normalisation.

    :param files: Files to be read from.
    :return: a tuple containing the raw data, its labels and the feature vectors.
    """
    question, answer, topic = "", "", ""
    for file in files:
        temp = pd.read_csv(file, sep='\t', header=None)

        question = temp[0]
        answer = temp[1]
        topic = temp[2]

    vectorizer = TfidfVectorizer(use_idf=True, analyzer='word', strip_accents='ascii', lowercase=True,
                                 stop_words='english')

    features = vectorizer.fit_transform(question)

    save(model_path + question_vectoriser_file, vectorizer)

    features = features.toarray()

    return question, answer, topic, features


def read_test_data(*files, state):
    """
    Function to read and vectorise raw test data from a list of given files.

    :param files: Files to be read from.
    :return: a tuple containing the raw data and the feature vectors.
    """
    data = []

    for file in files:
        temp = pd.read_csv(file, sep='\t', header=None)
        data.append(temp[0])

    data = pd.concat(data)

    if check_file(model_path + vectoriser_file, model_path + tfvectoriser_file):
        if state == 'n':
            vect = load(model_path + tfvectoriser_file)
        elif state == 'u':
            vect = load(model_path + vectoriser_file)

        features = vect.transform(data)

    return data, features


# In[ ]:


def regressor(x, y, _x, _y, state):
    """
    Function to train and save a logistic regression model on given data.

    :param x: Training data.
    :param y: Training data labels.
    :param _x: Testing data.
    :param _y: Testing data labels.
    :param state: a key to determine if the model is being trained on normalised ('n') or unnormalised ('u') data.
    :return: Predicted y values and accuracy results.
    """
    regressor = LogisticRegression()
    regressor = regressor.fit(X=x, y=y)
    y_pred = regressor.predict(_x)
    res = accuracy_score(_y, y_pred)

    if state == 'u':
        save(model_path + regressor_file, regressor)
    elif state == 'n':
        save(model_path + regressor_n_file, regressor)

    return y_pred, res


def NB_Classifier(x, y, _x, _y, state):
    """
    Function to train and save a naive bayes model on given data.

    :param x: Training data.
    :param y: Training data labels.
    :param _x: Testing data.
    :param _y: Testing data labels.
    :param state: a key to determine if the model is being trained on normalised ('n') or unnormalised ('u') data.
    :return: Predicted y values and accuracy results.
    """
    nbc = MultinomialNB()
    nbc = nbc.fit(X=x, y=y)
    y_pred = nbc.predict(_x)
    res = accuracy_score(_y, y_pred)

    if state == 'u':
        save(model_path + nbc_file, nbc)
    elif state == 'n':
        save(model_path + nbc_n_file, nbc)

    return y_pred, res


# In[ ]:


amazon = "data/amazon_cells_labelled.txt"
imdb = "data/imdb_labelled.txt"
yelp = "data/yelp_labelled.txt"


def train():
    """
    Function to train all possible model combinations.

    """
    questions, answers, topics, features = read_data_n(amazon, imdb, yelp)
    x_train, x_test, y_train, y_test = train_test_split(features, answers, test_size=0.2, random_state=0)

    pred, res = NB_Classifier(x_train, y_train, x_test, y_test, 'u')
    print("Unnormalised Linear Regression: {}%".format(round(res * 100, 3)))
    pred, res = NB_Classifier(x_train, y_train, x_test, y_test, 'u')
    print("Unnormalised Naive Bayes: {}%".format(round(res * 100, 3)))

    print("====================================")

    questions, answers, topics, features = read_data_n(amazon, imdb, yelp)
    x_train, x_test, y_train, y_test = train_test_split(features, answers, test_size=0.2, random_state=0)

    pred, res = regressor(x_train, y_train, x_test, y_test, 'n')
    print("Normalised Linear Regression: {}%".format(round(res * 100, 3)))
    pred, res = NB_Classifier(x_train, y_train, x_test, y_test, 'n')
    print("Normalised Naive Bayes: {}%".format(round(res * 100, 3)))


train()


# In[ ]:


def main(argv):
    """
    Main function to select the model to be tested on with new data from the console.

    :param argv: A sys.argv variable to take data from the console.
    """
    while not check_file(model_path + regressor_file,
                         model_path + regressor_n_file,
                         model_path + nbc_file,
                         model_path + nbc_n_file,
                         model_path + vectoriser_file,
                         model_path + tfvectoriser_file):
        print("Some files are unavailable. Retraining.")
        train()
        print('Retraining successful!')

    regressor = load(model_path + regressor_file)
    regressor_n = load(model_path + regressor_n_file)
    nbc = load(model_path + nbc_file)
    nbc_n = load(model_path + nbc_n_file)

    if argv[1] == "nb" and argv[2] == "u":
        data, features = read_test_data(data_path + argv[3], state="u")
        y = nbc.predict(features)
        write(y, res_path + "nb_u_results.txt")

    if argv[1] == "nb" and argv[2] == "n":
        data, features = read_test_data(data_path + argv[3], state="n")
        y = nbc_n.predict(features)
        write(y, res_path + "nb_n_results.txt")

    if argv[1] == "lr" and argv[2] == "u":
        data, features = read_test_data(data_path + argv[3], state="u")
        y = regressor.predict(features)
        write(y, res_path + "lr_u_results.txt")

    if argv[1] == "lr" and argv[2] == "n":
        data, features = read_test_data(data_path + argv[3], state="n")
        y = regressor_n.predict(features)
        write(y, res_path + "lr_n_results.txt")

    print("Success!\nPlease check `{}{}_{}_results.txt` for expected results".format(res_path, argv[1], argv[2]))


main(sys.argv)

# In[ ]:
