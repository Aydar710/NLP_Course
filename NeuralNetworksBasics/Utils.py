import collections
import itertools
import pickle

filters = ['.', ',', ':', ';', '!', '?', '(', ')', '-', "'", '»', '«']


def deserialize_normalized_reviews():
    pickle_in = open("normallized_list.pickle", "rb")
    normalized_reviews = pickle.load(pickle_in)
    return normalized_reviews


def deserialize_my_normalized_reviews():
    pickle_in = open("my_reviews_normalized.pickle", "rb")
    normalized_reviews = pickle.load(pickle_in)
    return normalized_reviews


def filter_reviews(all_reviews):
    for review in all_reviews:
        for word in review:
            if filters.__contains__(word) or len(word) == 1:
                review.remove(word)


def sort_dict(dict):
    ordered_dict = collections.OrderedDict(sorted(dict.items(), reverse=True, key=lambda t: t[1]))
    return ordered_dict


def get_500_most_frequent_words(ordered_dict):
    return dict(itertools.islice(ordered_dict.items(), 500))
