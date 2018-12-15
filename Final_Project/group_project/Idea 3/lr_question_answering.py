# coding: utf-8

# In[112]:


import re

import nltk
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# In[113]:


def read_data():
    question_data = pd.read_csv('Questions.txt', sep='\t', names=['questions'], index_col=None, header=0,
                                encoding="ISO-8859-1")
    answers_data = pd.read_csv('Answers.txt', sep='\t', names=['answers'], index_col=None, header=0,
                               encoding="ISO-8859-1")

    # After getting the data I combine them into 1 data frame to get questions and topics
    q_t_data = pd.concat([question_data, answers_data], axis=1)

    # I then return the questions and topics as a list
    return q_t_data.questions.values.tolist(), q_t_data.answers.values.tolist()


# In[114]:


def stemming(questions):
    lancaster_stemmer = LancasterStemmer()
    n_questions = []
    for question in questions:
        n_questions.append(' '.join(lancaster_stemmer.stem(token) for token in nltk.word_tokenize(question)))
    return (n_questions)


# In[115]:


def lemming(questions):
    wordnet_lemmatizer = WordNetLemmatizer()
    n_questions = []
    for question in questions:
        n_questions.append(' '.join(wordnet_lemmatizer.lemmatize(token) for token in nltk.word_tokenize(question)))
    return (n_questions)


# In[116]:


def lr_train(questions, labels):
    # N-gram model that comes up with features or occurances of words

    count_vect = CountVectorizer(stop_words='english')
    tf_transform = TfidfTransformer()

    X_train_counts = count_vect.fit_transform(r for r in questions)

    # This is to scale down impact of tokens that occur very frequently and arent
    # very informative.
    # Term Frequency time Inverse Document Frequency

    X_train_tf = tf_transform.fit_transform(X_train_counts)

    # splitting data into test and training sets
    X_train, X_test, Y_train, Y_test = train_test_split(X_train_tf, labels, random_state=35, train_size=0.80,
                                                        test_size=0.20)

    # MAKING THE MODEL
    lrn = LogisticRegression()
    model = lrn.fit(X_train, Y_train)
    # array.reshape(-1, 1)

    return lrn, tf_transform, count_vect, model, X_test, Y_test


# In[117]:


def lr_test(test_file, questions, labels):
    lrn, tf_transform, count_vect, model, X_test, Y_test = lr_train(questions, labels)
    predicted = model.predict(X_test)
    print("The accuracy of the model is: ", accuracy_score(Y_test, predicted))
    # print("This is a more detailed report of the QA models performance: \n", classification_report(Y_test, predicted))

    # Now using the test set I need to know if the predictions are accurate
    test_data = []
    for line in open(test_file):
        r2 = re.compile(r'[^a-zA-Z0-9]', re.MULTILINE)
        s = r2.sub(' ', line)
        test_data.append(s)
    test_data = stemming(test_data)
    test_data = lemming(test_data)

    # Making sentiment classification
    for i in range(len(test_data)):
        X_pred_counts = count_vect.transform(test_data)
        X_pred_tf = tf_transform.transform(X_pred_counts)
        predictions = lrn.predict(X_pred_tf)

    return (predictions, "\n")


# In[118]:


def question_LR(testfile):
    questions, answers = read_data()
    questions = stemming(questions)
    questions = lemming(questions)

    predicts = lr_test(testfile, questions, answers)
    results(predicts)


# In[119]:


def results(predicted):
    results = open("qa_results.txt", "w+")

    for label in predicted[0]:
        results.write(str(label) + "\n")

# In[120]:
