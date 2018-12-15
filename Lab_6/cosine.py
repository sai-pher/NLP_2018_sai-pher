import numpy as np


def baby_vectoriser(corpus):
    r = ""
    for i in corpus:
        r = r + " " + i

    r = r.split()

    data = []
    for i in r:
        if i not in data:
            data.append(i)

    return data


def to_array(a, data):
    _a = np.zeros(shape=len(data))

    for i in range(len(data)):
        if data[i] in a.split():
            _a[i] = 1

    return _a


def cos_sim(a, b):
    """Takes 2 vectors a, b and returns the cosine similarity according
    to the definition of the dot product
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)


# the counts we computed above
sentence_m = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0])
sentence_h = np.array([0, 0, 1, 1, 1, 1, 0, 0, 0])
sentence_w = np.array([0, 0, 0, 1, 0, 0, 1, 1, 1])

corpus = ["Mason really loves food", "Hannah loves food too", "The whale is food"]
voc = baby_vectoriser(corpus)
a = to_array(corpus[0], voc)
b = to_array(corpus[1], voc)
c = to_array(corpus[2], voc)

# We should expect sentence_m and sentence_h to be more similar
print(cos_sim(a, b))  # 0.5
print(cos_sim(a, c))  # 0.25

# full distin


corpus = ["How many teams participated in the first world cup?",
          "Who holds the record for top scorer in a single World Cup?", "Should I sell my car myself or trade it in?",
          "How important is car maintenance?", "How much does AWS SAM cost to use",
          "Which languages does AWS SAM support"]


def full(corpus):
    voc = baby_vectoriser(corpus)
    s = len(corpus)
    full = np.zeros(shape=[s, s])

    for i in range(s):
        for j in range(s):
            a = to_array(corpus[i], voc)
            b = to_array(corpus[j], voc)
            full[i][j] = cos_sim(a, b)
    return full


f = full(corpus)
print(f)
