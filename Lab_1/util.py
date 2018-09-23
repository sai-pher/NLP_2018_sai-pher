import random
import string
from math import *

import nltk
from nltk.stem.lancaster import LancasterStemmer


class NB_DataHandler:

    def __init__(self, *files: string, train_partition: float = 0.8, quiet: bool = True, shuffle: bool = True,
                 trim: bool = True):
        """
        Constructor for the Naive Bayes data class. Calls `self.data_cutter()`, `self.model_builder()`,
        `self.trim_common()`, `self.count_classes()`, `self.log_priors()` and `self.log_likelihood()`. This sets up
        the data and builds the Naive Bayes model inputted files.

        :type trim: bool
        :type shuffle: bool
        :type quiet: bool
        :type train_partition: float
        :type files: string
        :param files: Comma separated string objects of the path to each raw data file.
        :param train_partition: The decimal portion of the total data to be used for training.
            The remaining will be used for testing.
        :param quiet: Boolean trigger denoting whether function calls should hide console progress messages.
        :param shuffle: Boolean trigger denoting whether or not to shuffle data before partitioning.
        :param trim: Boolean trigger denoting whether or not to remove too common words from vocabulary.
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
        """
        Function to format raw data into training and testing format for classifier use.
        Partition is based on `self.partition` value.

        :rtype: (list, list, list, dict)
        :return: tuple containing total data, training data, test data and initialised document classes dictionary.
        """
        # local variable declarations
        files = self.files
        total_data = []
        training_data = []
        test_data = []
        doc_classes = {}
        punctuations = string.punctuation

        # File reading
        for file in files:
            data = open(file, "r")

            line_count = 0  # variable to count added lines
            word_count = 0  # variable to count added words

            # line reading
            for line in data:
                t_line = line \
                    .rstrip('\n') \
                    .split('\t')

                # line elements
                label = int(t_line[1])
                sentence = t_line[0].lower()

                # removes punctuations
                for sym in punctuations:
                    sentence = sentence.replace(sym, '')

                # adds new labels to doc_classes
                if label not in doc_classes:
                    doc_classes[label] = 0

                # tokenize and stem words in each sentence.
                stemmer = LancasterStemmer()
                words = nltk.word_tokenize(sentence)
                words = [stemmer.stem(word) for word in words]

                # wrap and add document to total_data.
                doc = (label, words)
                total_data.append(doc)

                word_count += len(words)
                line_count += 1

            # shuffle trigger to shuffle data.
            if self.shuffle:
                random.shuffle(total_data)

            # partition data.
            training_data = total_data[0:int(self.train_partition * len(total_data))]
            test_data = total_data[int(self.train_partition * len(total_data)):]

            # console readout.
            if not self.quiet:
                print('{} finished on line {} with {} words'
                      '\n==========================\n'.format(file, line_count, word_count))
            data.close()

        return total_data, training_data, test_data, doc_classes

    def model_builder(self):
        """
        Function to build the vocabulary and count the frequency of each word in the training data.

        :rtype: dict
        :return: A dictionary of unique words and their frequencies per class.
        """
        vocabulary = {}
        stop_words = []  # set(stopwords.words('english')) # optional stop word removal
        word_count = 0

        # access each document in the training data
        for doc in self.training_data:

            label = doc[0]
            words = doc[1]  # TODO: implement list(set(words)) for binary

            # count the frequency of each class
            if label not in self.doc_classes:
                self.doc_classes[label] = 1
            else:
                self.doc_classes[label] += 1

            # loop through each sentence and add and count each word in the vocabulary.
            for word in words:
                # set up for optional stop word removal
                if word in stop_words:
                    pass
                else:
                    # add new words to vocabulary
                    if word not in vocabulary:
                        # initialising word in vocabulary
                        vocabulary[word] = {"frequency": {l: 0 for l in self.doc_classes}, "class": [label]}
                        vocabulary[word]["frequency"][label] += 1
                    else:
                        # count words already in vocabulary
                        if label in vocabulary[word]["class"]:
                            vocabulary[word]["frequency"][label] += 1

                        else:
                            vocabulary[word]["class"].append(label)
                            vocabulary[word]["frequency"][label] = 1

                    word_count += 1

        # console readout
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
        """
        Function to count the number of words in each class.

        :rtype: dict
        :return: A dictionary of each class and the number of words in it.
        """
        voc_class = {l: 0 for l in self.doc_classes}
        for word, data in self.vocabulary.items():
            for cls in self.doc_classes:
                voc_class[cls] += data['frequency'][cls]
        return voc_class

    def log_likelihood(self):
        """
        Calculates the log likelihood of each word in the vocabulary.
        Adds these values to a dictionary object in the vocabulary under the key:
        `probability`
        """
        # loop through each word in vocabulary.
        for word in self.vocabulary:
            self.vocabulary[word]["probability"] = {}  # initialise 'probability' key for each word.

            # retrieve the frequency and class of each word.
            for cls, wfr in self.vocabulary[word]["frequency"].items():
                # calculate and add log probability to vocabulary
                cf = self.voc_classes[cls]

                p = log10((wfr + 1) / (cf + len(self.vocabulary)))

                self.vocabulary[word]["probability"][cls] = p

        # console readout
        if not self.quiet:
            print("Log probabilities calculated")

    def log_prior(self):
        """
        Calculates the log priors of each class in the data set.
        Creates a new dictionary to hold each class and its prior.

        :rtype: dict
        :return: A dictionary of each class and its log probability.
        """
        log_priors = {}  # initialise log priors dictionary
        n_doc = len(self.training_data)

        # loop through each class and its frequency
        for cls, n_c in self.doc_classes.items():
            # calculate log probability of classes
            pc = log10(n_c / n_doc)
            log_priors[cls] = pc

        return log_priors

    def nb_classifier(self, doc):
        """
        Main classification function. This calculated the Naive Bayes probability of a given sentence.

        :param doc: A sentence to be classified.
        :return: The predicted class of a given text.
        """
        sum_c = {}  # Initialise sum dictionary to hold sentence predicted values.

        # loop through priors and words in sentences to calculate the probability sums.
        for cls, lp in self.log_priors.items():
            # print('{} -> {}'.format(cls, lp))
            sum_c[cls] = lp
            for word in doc:
                if word in self.vocabulary:
                    sum_c[cls] += self.vocabulary[word]['probability'][cls]

        return max(zip(sum_c.values(), sum_c.keys()))

    def test(self):
        """
        Internal test and model validation function. classifies test data sentences
        and calculates the accuracy of the model.

        :return: The percentage score of the correctly classified cases over the total number of cases.
        """
        score = 0
        word_count = 0

        for doc in self.test_data:
            words = doc[1]
            label = doc[0]

            # Classify sentence and check if correctly classified.
            result = self.nb_classifier(words)
            if result[1] == label:
                score += 1
            else:

                # Add misclassified cases to report array for report generation.
                self.report_data.append(doc)

            word_count += 1

        return (score / word_count) * 100

    def trim_common(self):
        """
        Removes all words with class frequencies within 10% of each other.
        Uses only words above the a given count to preserve rare words.
        """
        # remove words with similar frequencies
        del_w = []
        for word, word_freq in self.vocabulary.items():
            if len(word_freq['class']) > 1:  # Select words in both or more than 1 class
                if sum(word_freq['frequency'].values()) > 60:
                    total = sum(word_freq['frequency'].values())
                    percentage = []
                    for label, label_freq in word_freq['frequency'].items():
                        percentage.append((label_freq / total) * 100)

                    # TODO: replace with functools.reduce() function for future proofing.
                    if len(percentage) == 2:
                        dif = abs(percentage[0] - percentage[1])
                    else:
                        # future proof
                        dif = 10
                    if dif < 10:
                        del_w.append(word)

        # remove common words from vocabulary
        for word in del_w:
            del self.vocabulary[word]

        # console readout
        if not self.quiet:
            print('{} words removed from vocabulary'.format(len(del_w)))
        del del_w

    def report(self, step=10):
        """
        Generates missclassification report in console. This returns each missclassified
        sentence and shows how each word was classified. This is t give a better idea how how the model
        is behaving at it's extremes.
        :param step:
        """
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
