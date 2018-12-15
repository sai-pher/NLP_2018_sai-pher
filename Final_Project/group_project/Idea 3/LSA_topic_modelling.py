# %%
import re
from random import random

from gensim import models, corpora
from gensim import similarities
from nltk import word_tokenize
from nltk.corpus import stopwords

train_percent = 0.80

questions_file = 'Questions.txt'
answers_file = 'Answers.txt'
topics_file = 'Topics.txt'

NUM_TOPICS = 200
TOP_N = 1
STOPWORDS = stopwords.words('english')


def remove_punctuations(s):
    return s.replace(',', '').replace('.', '').replace('!', '').replace('?', '').replace('(', '').replace(
        ')', '').replace('"', '').replace('$', '').replace("'", '').replace(':', '').replace('*', '').replace('%', '')


# training data
def clean(s):
    if s == '\n':
        return ''

    temp = s
    s = re.sub(r'^[0-9]+\.?\t? ?', '', s).rstrip().lower()
    if s == '':
        s = temp.rstrip()
    s = remove_punctuations(s)

    return s


def load_data():
    training_data = {'questions': [], 'answers': [], 'topics': []}
    test_data = {'questions': [], 'answers': [], 'topics': []}
    with open(questions_file) as q, open(answers_file) as a, open(topics_file) as t:
        questions = [clean(l) for l in q if clean(l) != '']
        answers = [clean(l) for l in a if clean(l) != '']
        topics = [clean(l) for l in t if clean(l) != '']

    questions.pop()
    topics.pop()

    for i in range(len(questions)):
        if random() < train_percent:
            training_data['questions'].append(questions[i])
            training_data['answers'].append(answers[i])
            training_data['topics'].append(topics[i])
        else:
            test_data['questions'].append(questions[i])
            test_data['answers'].append(answers[i])
            test_data['topics'].append(topics[i])

    return training_data, test_data


def argmax(topic_counts):
    max_topic = ''
    max_count = 0
    for topic in topic_counts:
        if topic_counts[topic] > max_count:
            max_count = topic_counts[topic]
            max_topic = topic

    return max_topic, max_count


def check_accuracy(test):
    correct_count = 0
    n = 0
    for i in range(len(test['questions'])):
        n += 1
        topic = classify_topic(test['questions'][i])

        print('predicted topic:', topic)
        print('actual topic:', test['topics'][i])

        if topic == test['topics'][i]:
            correct_count += 1

        print(str(correct_count) + '/' + str(n))
        print()

        # if n == 1:
        #     break

    return correct_count / n


def clean_text(text):
    tokenized_text = word_tokenize(text.lower())
    cleaned_text = [t for t in tokenized_text if t not in STOPWORDS and re.match('[a-zA-Z\-][a-zA-Z\-]{2,}', t)]
    return cleaned_text


def get_topic(id, topic_bounds):
    for topic in topic_bounds:
        if topic_bounds[topic][0] <= id <= topic_bounds[topic][1]:
            return topic


training_data, test_data = load_data()

print('training data:', len(training_data['questions']))
print('test data:', len(test_data['questions']))

# print('topic boundaries\n', topic_boundaries)

data = training_data['questions']

# For gensim we need to tokenize the data and filter out stopwords
tokenized_data = []
for text in data:
    tokenized_data.append(clean_text(text))

# Build a Dictionary - association word to numeric id
dictionary = corpora.Dictionary(tokenized_data)

# Transform the collection of texts to a numerical form
corpus = [dictionary.doc2bow(text) for text in tokenized_data]

# Build the LSI model
lsi_model = models.LsiModel(corpus=corpus, num_topics=NUM_TOPICS, id2word=dictionary)


# %%

def classify_topic(text):
    bow = dictionary.doc2bow(clean_text(text))

    # lsi_dist = lsi_model[bow]

    print("=" * 20)

    lsi_index = similarities.MatrixSimilarity(lsi_model[corpus])

    # Let's perform some queries
    mat_similarities = lsi_index[lsi_model[bow]]

    # Sort the similarities
    mat_similarities = sorted(enumerate(mat_similarities), key=lambda item: -item[1])

    # Top most similar documents:
    # similarities.sort(key=lambda t: t[0], reverse=False)
    print('similarities\n', mat_similarities[:10])
    print()
    print('question len:', len(training_data['questions']))
    print('answer len:', len(training_data['answers']))
    print('topic len:', len(training_data['topics']))
    print()

    # print()
    # print('doc check')
    # print(training_data['answers'][1260])

    # print all similar documents
    count = {}
    for document_id, similarity in mat_similarities[:TOP_N]:
        # print()
        # print('document id =', document_id)
        # print('question:', training_data['questions'][document_id])
        # print('answer:', training_data['answers'][document_id])

        doc_topic = training_data['topics'][document_id]
        # print('topic:', doc_topic)

        if doc_topic not in count:
            count[doc_topic] = 1
        else:
            count[doc_topic] += 1

        max_topic, max_count = argmax(count)

    print()
    print('most likely topic:', max_topic)

    # print('topic counts:\n', count)

    return max_topic


# if __name__ == '__main__':

for i in range(20):
    accuracy = check_accuracy(test_data)
    print('accuracy =', accuracy)

    with open('accuracies.txt', 'a+') as f:
        print(NUM_TOPICS, TOP_N, accuracy, file=f)

    TOP_N += 5
