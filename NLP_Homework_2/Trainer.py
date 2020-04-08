from __future__ import division

import pickle
from collections import defaultdict
import pymorphy2
import nltk
import pandas


def train(sample):
    morph = pymorphy2.MorphAnalyzer()
    classes, freq = defaultdict(int), defaultdict(int)
    for row in sample.itertuples():
        classes[row.label] += 1

        for word in nltk.word_tokenize(row.text):
            if word.__len__() > 1:
                parsed_word = morph.parse(word)[0]
                feat = parsed_word.normal_form
                freq[row.label, feat] += 1

        print(sample.__len__() - row.Index)
    for label, feat in freq:
        freq[label, feat] /= classes[label]
    for c in classes:
        classes[c] /= len(sample)

    return classes, freq


def serialize_classifier(classifier):
    pickle_out = open("classifier.pickle", "wb")
    pickle.dump(classifier, pickle_out)
    pickle_out.close()


reviews = pandas.read_csv('reviews.csv', delimiter=',')
classifier = train(reviews[['text', 'label']])
serialize_classifier(classifier)
