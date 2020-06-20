from nltk import ngrams
# import numpy as np
from collections import Counter


def get_grams_l(n, l):
    labels = range(7)

    # n = 6
    # top = 50

    grams = []

    for label in labels:
        with open(str(label)+'_train.txt') as f:
            c = f.read()
        grams_bytes = ngrams(''.join(c.split()), n)
        # grams_words = ngrams(c.split(), n)
        gram_counter = Counter(grams_bytes)
        # bla_split = Counter(grams_words)
        top_grams = []
        for gram,_ in gram_counter.most_common(l):
            top_grams.append(gram)
        grams.append(top_grams)

    return grams


def get_grams(n):
    labels = range(7)

    # n = 6
    # top = 50

    grams = []

    for label in labels:
        with open(str(label)+'_train.txt') as f:
            c = f.read()
        # grams_bytes = ngrams(c, n)
        grams_bytes = ngrams(''.join(c.split()), n)
        grams.append(Counter(grams_bytes))

    return grams