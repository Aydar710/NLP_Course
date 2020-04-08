from __future__ import division
import pickle
from collections import defaultdict
from math import log
import pandas
import pymorphy2
import nltk


def classify(classifier, feats):
    classes, prob = classifier
    return min(classes.keys(),
                     key=lambda cl: -log(classes[cl]) + sum(-log(prob.get((cl, feat), 10 ** (-7))) for feat in feats))


def classify_reviews(reviews):
    true_result = 0
    false_result = 0

    for row in reviews.itertuples():
        morph = pymorphy2.MorphAnalyzer()
        words = nltk.word_tokenize(row.text)
        feats = []
        for word in words:
            feats.append(morph.parse(word)[0].normal_form)

        classified = classify(classifier, feats)
        if classified == row.label:
            true_result += 1
        else:
            false_result += 1
    print("true results = " + str(true_result))
    print("false results = " + str(false_result))
    accuracy = (true_result / reviews.__len__()).__round__(2)
    print("Accuracy = " + str(accuracy))


pickle_in = open("classifier.pickle", "rb")
classifier = pickle.load(pickle_in)

my_reviews = pandas.read_csv('my_reviews.csv', delimiter=',')
all_reviews = pandas.read_csv('reviews.csv', delimiter=',')

print("my_reviews:")
classify_reviews(my_reviews)

print()

print("all_reviews:")
classify_reviews(all_reviews)
