import random


def data_cutter(*file: str):
    """

    :type file: str
    """
    # code to cut data
    total_data = []
    training_data = []
    test_data = []
    classes = {}
    vocabulary = {}

    word_count = 0

    for f in file:
        data = open(f, "r")

        c = 0
        w = 0

        for line in data:
            t_line = line \
                .rstrip('\n') \
                .split('\t')

            label = t_line[1]
            sentence = t_line[0]

            # split words in each sentence. future: tokenize and remove filler words
            words = sentence.split()

            doc = (label, words)

            total_data.append(doc)

            w += len(words)

        random.shuffle(total_data)

        training_data = total_data[0:int(0.8 * len(total_data))]
        test_data = total_data[int(0.8 * len(total_data)):]

        c += 1

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
                vocabulary[word] = {"frequency": {label: 1}, "class": [label]}
            else:
                if label in vocabulary[word]["class"]:
                    vocabulary[word]["frequency"][label] += 1

                else:
                    vocabulary[word]["class"].append(label)
                    vocabulary[word]["frequency"][label] = 1

            if label not in classes:
                classes[label] = 1
            else:
                classes[label] += 1

            word_count += 1
    #
    #     c += 1
    #
    # print('{} finished on line {} with {} words'
    #       '\n==========================\n'.format(f, c, w))

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

    return total_data, training_data, test_data, classes, vocabulary
