import pickle
from collections import defaultdict
import nltk
import numpy
import pymorphy2
import pandas
import keras
from keras import Sequential
from keras.layers import Dense, Activation, Dropout
from keras_preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline

from Utils import deserialize_my_normalized_reviews, filter_reviews, sort_dict, get_500_most_frequent_words

reviews = deserialize_my_normalized_reviews()

filter_reviews(reviews)

frequency = defaultdict(int)

for review in reviews:
    for word in review:
        frequency[word] += 1

ordered_frequency = sort_dict(frequency)
most_frequent_dict = get_500_most_frequent_words(ordered_frequency)
words_used = list(most_frequent_dict.keys())

reviews_with_spaces = []
for review in reviews:
    reviews_with_spaces.append(" ".join(review))

pipe = Pipeline([('count', CountVectorizer(vocabulary=words_used)), ('tfid', TfidfTransformer())]).fit(
    reviews_with_spaces)

# TF-IDF's
tf_idf = pipe['tfid'].idf_

# get x_train
x_train = []
for i in range(0, len(words_used)):
    train_list = []
    for k in range(0, len(reviews)):
        if words_used[i] in reviews[k]:
            train_list.append(tf_idf[i])
        else:
            train_list.append(0)
    x_train.append(train_list.copy())
    train_list.clear()

tokenizer = Tokenizer(num_words=500)
x_train = tokenizer.sequences_to_matrix(x_train, mode='tfidf')

# get y_train
df = pandas.read_csv('normallized_reviews.csv', delimiter=',')
y_train = df.label.values
y_train = keras.utils.to_categorical(y_train, 3)

# building model
model = Sequential()
model.add(Dense(500, input_shape=(500, 32)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(3))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.metrics_names)

history = model.fit(x_train, y_train, batch_size=32, epochs=2, verbose=1, validation_split=0.1)
score = model.evaluate(x_train, y_train, batch_size=32, verbose=1)

print("Test loss: {}".format(score[0]))
print("Test accuracy: {}".format(score[1]))
