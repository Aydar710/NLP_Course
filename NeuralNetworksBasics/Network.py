import keras
from keras import Sequential
from keras.layers import Dense, Dropout
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas
from sklearn.metrics import classification_report

from Utils import deserialize_my_normalized_reviews, deserialize_normalized_reviews

# Reviews for training model, received from pickle
reviews_to_train = deserialize_normalized_reviews()

all_reviews = pandas.read_csv('normallized_reviews.csv', delimiter=',')
y_test = pandas.read_csv('my_labels.csv', delimiter=',')

# get training reviews
reviews_training = []
for review in reviews_to_train:
    reviews_training.append(" ".join(review))

# get testing reviews
my_reviews = deserialize_my_normalized_reviews()
reviews_testing = []
for review in my_reviews:
    reviews_testing.append(" ".join(review))

# tf-idf
tfidf = TfidfVectorizer(max_features=500)
tfidf.fit_transform(reviews_training)
x_train_tfidf = tfidf.fit_transform(reviews_training)
x_test_tfidf = tfidf.fit_transform(reviews_testing)

# building model
y_train_categorical = keras.utils.to_categorical(all_reviews["label"], 3)
y_test_categorical = keras.utils.to_categorical(y_test, 3)

model = Sequential()
model.add(Dense(512, input_shape=(500,)))
model.add(Dropout(0.5))
model.add(Dense(3))
model.compile(metrics=["accuracy"], optimizer='adam', loss='categorical_crossentropy')

# Fitting model
model.fit(x_train_tfidf, y_train_categorical, epochs=10, batch_size=32)

# Testing model
result = model.predict(x_test_tfidf)

print(classification_report(y_test_categorical.argmax(axis=1), result.argmax(axis=1)))
