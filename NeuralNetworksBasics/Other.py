import keras
from keras import Sequential
from keras.layers import Dense, Dropout
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas
from sklearn.metrics import classification_report

from Utils import deserialize_my_normalized_reviews, deserialize_normalized_reviews

tfidf = TfidfVectorizer(max_features=500)

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
tfidf.fit_transform(reviews_training)
Train_X_Tfidf = tfidf.fit_transform(reviews_training)
Test_X_Tfidf = tfidf.fit_transform(reviews_testing)

# building model
Train_Y_keras = keras.utils.to_categorical(all_reviews["label"], 3)
Test_Y_keras = keras.utils.to_categorical(y_test, 3)

model = Sequential()
model.add(Dense(512, input_shape=(500,)))
model.add(Dropout(0.5))
model.add(Dense(3))
model.compile(metrics=["accuracy"], optimizer='adam', loss='categorical_crossentropy')

model.fit(Train_X_Tfidf, Train_Y_keras, epochs=1, batch_size=32)

result = model.predict(Test_X_Tfidf)

print(classification_report(Test_Y_keras.argmax(axis=1), result.argmax(axis=1)))
