"""
===================================================
     Introduction to Machine Learning (67577)
             IML HACKATHON, June 2018

            **  Code Classifier  **

Auther(s):
Omri Ben Dov
Or Malka
===================================================
"""
from nltk import ngrams
from collections import Counter
import numpy as np


class Classifier(object):
    def __init__(self, toTrain=False):
        # to_Train - whether to train on new data, or load from ready file
        self.ns = (14, 6)
        self.file_name = 'ngrams.npz'
        self.l = 50000
        self.ngrams = []
        self.labels = range(7)
        self.DEBUG = False
        if toTrain:
            self.train()
        self.load()

    def predict(self, lines):
        keep_indices = range(7)
        for i, n in enumerate(self.ns):
            grams_bytes = ngrams(''.join(lines.split()), n)
            grams_bytes = set(grams_bytes)
            intersections = -1 * np.ones(7)
            for j, g in enumerate(self.ngrams[i]):
                if j in keep_indices:
                    intersections[j] = len(grams_bytes.intersection(g))
            max_intersection = np.amax(intersections)
            keep_indices = [j for j in keep_indices if intersections[j] == max_intersection]
            if len(keep_indices) == 1:  # check there is no tie
                return keep_indices[0]
            if self.DEBUG:
                print('TIE with n=: ', n, ' intersections= ', intersections)
        return keep_indices[0]

    def load(self):
        self.ngrams = np.load(self.file_name)['a']

    def get_grams(self, n, l):
        grams = []

        for label in self.labels:
            with open(str(label) + '_train.txt') as f:
                c = f.read()
            grams_bytes = ngrams(''.join(c.split()), n)
            temp_grams = []
            for g, _ in Counter(grams_bytes).most_common(l):
                temp_grams.append(g)
            grams.append(temp_grams)

        return grams

    def train(self):
        for n in self.ns:
            self.ngrams.append(self.get_grams(n, self.l))
        np.savez_compressed(self.file_name, a=self.ngrams)

    def classify(self, X):
        """
        Recieves a list of m unclassified pieces of code, and predicts for each
        one the Github project it belongs to.
        :param X: A list of length m containing the code segments (strings)
        :return: y_hat - a list where each entry is a number between 0 and 8
        0 - Sonar
        1 - Dragonfly
        2 - tensorflow
        3 - devilution
        4 - flutter
        5 - react
        6 - spritejs
        """
        predictions = []
        for i, code in enumerate(X):
            predictions.append(self.predict(code))
            if self.DEBUG and i % 100 == 0:
                print(str(i) + '/' + str(len(X)))
        return predictions
