# coding: utf-8

# #  Natural Language Processing Group Project
# ## Topic Modelling - Logistic Regression Implementation
# 
# Name: Our Group
# 

# In[74]:


import nltk

nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, classification_report
import sys


# In[75]:


def read_data():
    question_data = pd.read_csv('Questions.txt', sep='\t', names=['questions'], index_col=None, header=0,
                                encoding="ISO-8859-1")
    topics_data = pd.read_csv('Topics.txt', sep='\t', names=['topics'], index_col=None, header=0, encoding="ISO-8859-1")

    # After getting the data I combine them into 1 data frame to get questions and topics
    q_t_data = pd.concat([question_data, topics_data], axis=1)

    # I then return the questions and topics as a list
    return q_t_data.questions.values.tolist(), q_t_data.topics.values.tolist()


# In[76]:


# I used stemming only on questions to remove any unncessary words. 
def stemming(questions):
    lancaster_stemmer = LancasterStemmer()
    n_questions = []
    for question in questions:
        n_questions.append(' '.join(lancaster_stemmer.stem(token) for token in nltk.word_tokenize(question)))
    return (n_questions)


# In[77]:


def lemming(questions):
    wordnet_lemmatizer = WordNetLemmatizer()
    n_questions = []
    for question in questions:
        n_questions.append(' '.join(wordnet_lemmatizer.lemmatize(token) for token in nltk.word_tokenize(question)))
    return (n_questions)


# In[78]:


def lr_train(questions, labels):
    # N-gram model that comes up with features or occurances of words

    count_vect = CountVectorizer(stop_words='english')
    tf_transform = TfidfTransformer()

    X_train_counts = count_vect.fit_transform(r for r in questions)

    # This is to scale down impact of tkens that occur very frequently and arent
    # very informative.
    # Term Frequency time Inverse Document Frequency

    X_train_counts = count_vect.fit_transform(r for r in questions)

    # This is to scale down impact of tkens that occur very frequently and arent
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


# In[79]:


def lr_test(test_file, questions, labels):
    lrn, tf_transform, count_vect, model, X_test, Y_test = lr_train(questions, labels)

    # Now using the test set I need to know if I'm right
    # test_data = []
    # for line in open(test_file):
    #    r2 = re.compile(r'[^a-zA-Z0-9]', re.MULTILINE)
    #    s = r2.sub(' ', line)
    #    test_data.append(s)

    # test_data = stemming(test_data)
    # test_data = lemming(test_data)

    # Making sentiment classification
    # for i in range (len(test_data)):
    # X_pred_counts = count_vect.transform(test_data)
    # X_pred_tf = tf_transform.transform(X_pred_counts)
    # predictions = lrn.predict(X_pred_tf)

    # Evaluating the model using the classification report function form sklearn

    predicted = model.predict(X_test)
    print("The accuracy of this test is: ", accuracy_score(Y_test, predicted) * 100)

    print("This is a more detailed report of the classifiers performance: \n", classification_report(Y_test, predicted))

    return (predicted, "\n")


# In[80]:


def my_LR(testfile):
    questions, topics = read_data()
    questions = stemming(questions)
    questions = lemming(questions)
    lr_test(testfile, questions, topics)


# In[81]:


def results(classifier, version, predictL):
    file_name = open("results-" + classifier + "-" + version + ".txt", w)
    # file_name.write("Ouput: "+"\n")

    for label in predictL[0]:
        file_name.write(str(label) + "\n")


# In[82]:


def main():
    script = sys.argv[0]
    file_name = sys.argv[1]
    my_LR()


main()
