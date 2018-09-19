import random
import string
from math import *

import nltk
from nltk.stem.lancaster import LancasterStemmer


class NB_DataHandler:

    def __init__(self, *files, train_partition=0.8, quiet=True, shuffle=True, trim=True):
        """
        Constructor for the Nive Bayes data class
        :param files:
        :param train_partition:
        :param quiet:
        :param shuffle:
        :param trim:
        """
        self.files = files
        self.train_partition = train_partition
        self.quiet = quiet
        self.shuffle = shuffle
        self.trim = trim
        self.report_data = []

        self.total_data, self.training_data, self.test_data, self.doc_classes = self.data_cutter()
        self.vocabulary = self.model_builder()
        if self.trim:
            self.trim_common()
        self.voc_classes = self.count_class()
        self.log_priors = self.log_prior()
        self.log_likelihood()

    def data_cutter(self):

        # code to cut data
        files = self.files
        total_data = []
        training_data = []
        test_data = []

        doc_classes = {}

        punct = string.punctuation

        for f in files:
            data = open(f, "r")

            c = 0
            w = 0

            for line in data:
                t_line = line \
                    .rstrip('\n') \
                    .split('\t')

                label = int(t_line[1])
                sentence = t_line[0].lower()

                for sym in punct:
                    sentence = sentence.replace(sym, '')

                if label not in doc_classes:
                    doc_classes[label] = 0

                # split words in each sentence.
                # TODO: tokenize and remove filler words
                stemmer = LancasterStemmer()
                # words = sentence.split()
                words = nltk.word_tokenize(sentence)
                words = [stemmer.stem(word) for word in words]

                doc = (label, words)

                total_data.append(doc)

                w += len(words)
            # TODO: create shuffle trigger
            if self.shuffle:
                random.shuffle(total_data)

            # TODO: create portioning variable
            training_data = total_data[0:int(self.train_partition * len(total_data))]
            test_data = total_data[int(self.train_partition * len(total_data)):]

            c += 1
            if not self.quiet:
                print('{} finished on line {} with {} words'
                      '\n==========================\n'.format(f, c, w))
            data.close()

            # exit file loop

        return total_data, training_data, test_data, doc_classes

    def model_builder(self):

        # doc_classes = {}
        vocabulary = {}
        stop_words = []  # set(stopwords.words('english'))
        word_count = 0

        for doc in self.training_data:

            label = doc[0]
            words = doc[1]

            if label not in self.doc_classes:
                self.doc_classes[label] = 1
            else:
                self.doc_classes[label] += 1

            for word in words:
                if word in stop_words:
                    pass
                else:
                    if word not in vocabulary:
                        # initialising word in vocabulary
                        vocabulary[word] = {"frequency": {l: 0 for l in self.doc_classes}, "class": [label]}
                        vocabulary[word]["frequency"][label] += 1
                    else:
                        if label in vocabulary[word]["class"]:
                            vocabulary[word]["frequency"][label] += 1

                        else:
                            vocabulary[word]["class"].append(label)
                            vocabulary[word]["frequency"][label] = 1

                    word_count += 1

        if not self.quiet:
            print(
                "=====Summary=====\n"
                "Unique words:\t -> {}\n"
                "Total words:\t -> {}\n"
                "Total data:\t -> {}\n"
                "Train data:\t -> {}\n"
                "Test data:\t -> {}\n"
                    .format(len(vocabulary),
                            word_count,
                            len(self.total_data),
                            len(self.training_data),
                            len(self.test_data)))

        return vocabulary

    def count_class(self):
        nc = {l: 0 for l in self.doc_classes}
        for word, data in self.vocabulary.items():
            for cls in self.doc_classes:
                nc[cls] += data['frequency'][cls]
        return nc

    def log_likelihood(self):

        for word in self.vocabulary:
            self.vocabulary[word]["probability"] = {}
            for cls, wfr in self.vocabulary[word]["frequency"].items():
                cf = self.voc_classes[cls]

                p = log10((wfr + 1) / (cf + 1))

                self.vocabulary[word]["probability"][cls] = p
        if not self.quiet:
            print("Log probabilities calculated")

        return self

    def log_prior(self):
        log_priors = {}
        n_doc = len(self.training_data)
        for cls, n_c in self.doc_classes.items():
            pc = log10(n_c / n_doc)
            log_priors[cls] = pc

        return log_priors

    def nb_classifier(self, doc):
        sum_c = {}
        for cls, lp in self.log_priors.items():
            # print('{} -> {}'.format(cls, lp))
            sum_c[cls] = lp
            for word in doc:
                if word in self.vocabulary:
                    sum_c[cls] += self.vocabulary[word]['probability'][cls]

        return max(zip(sum_c.values(), sum_c.keys()))

    def test(self):
        score = 0
        word_count = 0

        for doc in self.test_data:
            words = doc[1]
            label = doc[0]

            result = self.nb_classifier(words)
            if result[1] == label:
                score += 1
            else:
                self.report_data.append(doc)

            word_count += 1

        return (score / word_count) * 100

    def trim_common(self):
        # remove words with similar frequencies
        del_w = []
        for w, fr in self.vocabulary.items():
            if len(fr['class']) > 1:
                if sum(fr['frequency'].values()) > 60:
                    tot = sum(fr['frequency'].values())
                    percentage = []
                    for c, f in fr['frequency'].items():
                        percentage.append((f / tot) * 100)
                    if len(percentage) == 2:
                        dif = abs(percentage[0] - percentage[1])
                    else:
                        # future proof
                        dif = 10
                    if dif < 10:
                        del_w.append(w)

        for w in del_w:
            del self.vocabulary[w]

        if not self.quiet:
            print('{} words removed from vocabulary'.format(len(del_w)))
        del del_w

    def report(self, step=10):
        word_count = 0
        i = 0
        for doc in self.report_data:
            words = doc[1]
            label = doc[0]

            result = self.nb_classifier(words)
            if result[1] != label:
                if i % step == 0:
                    print("sentence " + str(word_count))
                    print('sentence:\n{}\n\npredicted\t: {}\nactual\t: {}'.format(str(words), result[1], label))

                    for word in words:
                        if word in self.vocabulary:
                            p = self.vocabulary[word]['probability']
                            max_p = max(zip(p.values(), p.keys()))
                            print('{}\t-> {}'.format(word, max_p[1]))
                        else:
                            print('{}\t-> {}'.format(word, 'Nan'))
                    print('\n')
                i += 1
            word_count += 1
