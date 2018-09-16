import random
from math import *


class Data:

    def __init__(self, *files, quiet=True):
        """

        :param files:
        """
        self.files = files
        self.quiet = quiet
        self.total_data, \
        self.training_data, \
        self.test_data, \
        self.voc_classes, \
        self.doc_classes, \
        self.vocabulary = self.data_cutter()
        self.log_priors = self.log_prior()
        self.log_likelihood()

    def data_cutter(self):

        # code to cut data
        files = self.files
        total_data = []
        training_data = []
        test_data = []
        voc_classes = {}
        doc_classes = {}
        vocabulary = {}

        word_count = 0

        for f in files:
            data = open(f, "r")

            c = 0
            w = 0

            for line in data:
                t_line = line \
                    .rstrip('\n') \
                    .split('\t')

                label = int(t_line[1])
                sentence = t_line[0]

                if label not in doc_classes:
                    doc_classes[label] = 1
                else:
                    doc_classes[label] += 1

                # split words in each sentence.
                # TODO: tokenize and remove filler words
                words = sentence.split()

                doc = (label, words)

                total_data.append(doc)

                w += len(words)
            # TODO: create shuffle trigger
            random.shuffle(total_data)

            # TODO: create portioning variable
            training_data = total_data[0:int(0.8 * len(total_data))]
            test_data = total_data[int(0.8 * len(total_data)):]

            c += 1
            if not self.quiet:
                print('{} finished on line {} with {} words'
                      '\n==========================\n'.format(f, c, w))
            data.close()

            # exit file loop

        for doc in training_data:

            label = doc[0]
            words = doc[1]

            for word in words:
                if word not in vocabulary:
                    # initialising word in vocabulary
                    vocabulary[word] = {"frequency": {l: 0 for l in doc_classes}, "class": [label]}
                    vocabulary[word]["frequency"][label] += 1
                else:
                    if label in vocabulary[word]["class"]:
                        vocabulary[word]["frequency"][label] += 1

                    else:
                        vocabulary[word]["class"].append(label)
                        vocabulary[word]["frequency"][label] = 1

                if label not in voc_classes:
                    voc_classes[label] = 1
                else:
                    voc_classes[label] += 1

                word_count += 1

        #
        #     c += 1
        #
        # print('{} finished on line {} with {} words'
        #       '\n==========================\n'.format(f, c, w))
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
                            len(total_data),
                            len(training_data),
                            len(test_data)))

        return total_data, training_data, test_data, voc_classes, doc_classes, vocabulary

    # def vocabulary(self):
    #     return self.vocabulary
    #
    # def word(self, word):
    #     return self.vocabulary[word]

    def log_likelihood(self):

        for word in self.vocabulary:
            self.vocabulary[word]["probability"] = {}
            for cls, wfr in self.vocabulary[word]["frequency"].items():
                cf = self.voc_classes[cls]

                p = log10((wfr + 1) / (cf + len(self.vocabulary)))

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
            sum_c[cls] = lp
            for word in doc:
                if word in self.vocabulary:
                    sum_c[cls] += self.vocabulary[word]['probability'][cls]
        return max(sum_c.items())

    def test(self):
        score = 0
        word_count = 0
        for doc in self.test_data:
            words = doc[1]
            label = doc[0]

            result: tuple = self.nb_classifier(words)
            if result[0] == label:
                score += 1
            word_count += 1
        return (score / word_count) * 100
