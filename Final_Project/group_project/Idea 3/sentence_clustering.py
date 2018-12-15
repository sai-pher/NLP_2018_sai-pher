# %%
import re

# from sklearn.cluster import Birch
# from sklearn import metrics
import nltk
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# import numpy as np
# from nltk.cluster import KMeansClusterer


questions_file = 'Questions.txt'
answers_file = 'Answers.txt'
topics_file = 'Topics.txt'

all_topics = {}
topic_count = {}


def do_normalization(text):
    stemmer = nltk.LancasterStemmer()
    lemmatizer = nltk.WordNetLemmatizer()

    # removing this line improve the performance of the classifier after normalization
    text = text.lower()

    text = ' '.join([stemmer.stem(s) for s in text.split(' ')])
    text = ' '.join([lemmatizer.lemmatize(s) for s in text.split(' ')])

    text = remove_punctuations(text)

    return text


def remove_punctuations(s):
    return s.replace(',', '').replace('.', '').replace('!', '').replace('?', '').replace('(', '').replace(
        ')', '').replace('"', '').replace('$', '').replace("'", '').replace(':', '').replace('*', '').replace('%', '')


# training data
def clean(s):
    s = re.sub(r'[0-9]*\.*\t*', '', s).rstrip().lower()
    s = remove_punctuations(s)
    return s


def load_data():
    with open(questions_file) as q, open(answers_file) as a, open(topics_file) as t:
        questions = [clean(l) for l in q]
        answers = [clean(l) for l in a]
        topics = [clean(l) for l in t]

    return questions, answers, topics


questions, answers, topics = load_data()

documents = []
n = 0
for question, topic in zip(questions, topics):
    if topic not in all_topics:
        all_topics[topic] = len(all_topics) + 1
    if topic not in topic_count:
        topic_count[topic] = 1
    else:
        topic_count[topic] += 1

    documents.append(TaggedDocument(question, [n]))
    n += 1

# topics = enumerate(set(topics))
# print(list(topics))

# print(common_texts)

# documents = [TaggedDocument(doc, [i%3]) for i, doc in enumerate(common_texts)]
# %%

print('topics\n', all_topics)
print('topic counts')
for x in topic_count:
    print(x, topic_count[x])

model = Doc2Vec(documents, vector_size=5, window=10, min_count=1, workers=4, dm=2)

# %%
new_sentence = 'what crops are grown in ghana'
similars = model.docvecs.most_similar(positive=[model.infer_vector(new_sentence.lower().split())], topn=3)
print(similars)

print('similars', similars)
print()
for i, j in similars:
    print(questions[i])
    print(topics[i])
    print()

# start_alpha = 0.01
# infer_epoch = 1000

# X = []
# for t in questions:
#     X.append(model.infer_vector(t, alpha=start_alpha, steps=infer_epoch))
