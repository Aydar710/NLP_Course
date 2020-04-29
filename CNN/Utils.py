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


def serialize_ruscorpora_data(data):
    pickle_out = open("data/ruscorpora_data.pickle", "wb")
    pickle.dump(data, pickle_out)
    pickle_out.close()


def deserialize_ruscorpora_data():
    pickle_in = open("data/data.pickle", "rb")
    data = pickle.load(pickle_in)
    return data


def create_embedding_matrix(dataframe, word_index, dim):
    size = len(word_index) + 1
    matrix = np.zeros((size, dim))
    for index in range(len(dataframe["word"])):
        if dataframe["word"][index] in word_index:
            x = word_index[dataframe["word"][index]]
            matrix[x] = np.array(dataframe["value"][index], dtype=np.float32)[:dim]
    return matrix


def split_ruscorpora_data(ruscorpora_data):
    i = 0
    for row in ruscorpora_data["text"]:
        ruscorpora_data["text"][i] = row.split("_", 1)[0]
        i += 1
    i = 0
    for row in ruscorpora_data["value"]:
        ruscorpora_data["value"][i] = row.split(" ")
        i += 1
