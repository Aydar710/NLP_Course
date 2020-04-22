import pickle
import numpy as np


def deserialize_normalized_reviews():
    pickle_in = open("data/normallized_list.pickle", "rb")
    normalized_reviews = pickle.load(pickle_in)
    return normalized_reviews


def deserialize_my_normalized_reviews():
    pickle_in = open("data/my_reviews_normalized.pickle", "rb")
    normalized_reviews = pickle.load(pickle_in)
    return normalized_reviews


def serialize_model(model):
    pickle_out = open("data/Word2vecModel.pickle", "wb")
    pickle.dump(model, pickle_out)
    pickle_out.close()


def deserialize_model():
    pickle_in = open("data/Word2vecModel.pickle", "rb")
    model = pickle.load(pickle_in)
    return model


def deserialize_my_normalized_reviews():
    pickle_in = open("data/my_reviews_normalized.pickle", "rb")
    normalized_reviews = pickle.load(pickle_in)
    return normalized_reviews


def get_average_feature_vecs(reviews, model, num_features):
    count = 0
    feature_vecs = np.zeros((len(reviews), num_features), dtype='float32')
    for review in reviews:
        feature_vecs[count] = get_feature_vecs(review, model, num_features)
        count = count + 1
    return feature_vecs


def get_feature_vecs(words, model, features_num):
    feature_vec = np.zeros((features_num,), dtype="float32")
    word_count = 0
    index2word_set = set(model.wv.index2word)
    for word in words:
        if word in index2word_set:
            word_count = word_count + 1
            feature_vec = np.add(feature_vec, model[word])

    feature_vec = np.divide(feature_vec, word_count)
    return feature_vec
