import keras
import pandas
from keras.layers import Dense, Activation, Embedding, Conv1D, GlobalMaxPool1D
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import classification_report
from Utils import deserialize_normalized_reviews, deserialize_my_normalized_reviews, create_embedding_matrix, \
    serialize_ruscorpora_data, split_ruscorpora_data, deserialize_ruscorpora_data

batch_size = 32

# get training reviews
reviews_to_train = deserialize_normalized_reviews()
all_reviews = pandas.read_csv('data/normallized_reviews.csv', delimiter=',')

# get testing reviews
reviews_to_test = deserialize_my_normalized_reviews()
test_labels = pandas.read_csv('data/my_labels.csv', delimiter=',')

# get vectors from ruscorpora model
ruscorpora_data = pandas.read_csv('data/model.txt', skiprows=1, sep=r'\s{2,}', engine='python', names=[1])
ruscorpora_data = pandas.DataFrame(ruscorpora_data[1].str.split(r'\s{1,}', 1), columns=[1])
ruscorpora_data = ruscorpora_data[1].apply(pandas.Series)
ruscorpora_data.columns = ["text", "value"]

print("0")
# Use deserialized data further
split_ruscorpora_data(ruscorpora_data)

print("1")

serialize_ruscorpora_data(ruscorpora_data)
# ruscorpora_data = deserialize_ruscorpora_data()

tokenizer = Tokenizer(num_words=ruscorpora_data.size)
tokenizer.fit_on_texts(ruscorpora_data["text"])

# making reviews as sequences
reviews_to_test = tokenizer.texts_to_sequences(reviews_to_test)
reviews_to_train = tokenizer.texts_to_sequences(reviews_to_train)

print("2")

max_length = 0
for review in reviews_to_train.append(reviews_to_test):
    len_review = len(review)
    if len_review > max_length:
        max_length = len_review

# preparing reviews
reviews_train_prepared = pad_sequences(reviews_to_test, maxlen=max_length, padding='post')
reviews_test_prepared = pad_sequences(reviews_to_train, maxlen=max_length, padding='post')

# get categorical
labels_train_prepared = keras.utils.to_categorical(test_labels, 3)
labels_test_prepared = keras.utils.to_categorical(all_reviews["label"], 3)

# create emdedding matrix
embedding_matrix = create_embedding_matrix(ruscorpora_data, tokenizer.word_index, max_length)


# Building neural network
model = Sequential()
model.add(Embedding(ruscorpora_data.size, weights=[embedding_matrix], input_length=max_length,
                    trainable=False))
model.add(Conv1D(max_length, 3))
model.add(Activation("relu"))
model.add(GlobalMaxPool1D())
model.add(Dense(3))
model.add(Activation('softmax'))
model.compile(metrics=["accuracy"], optimizer='adam', loss='binary_crossentropy')

# fitting model
model.fit(reviews_train_prepared, labels_train_prepared, epochs=30, verbose=False)
# model.fit(reviews_train_prepared, labels_train_prepared, epochs=20, verbose=False)
# model.fit(reviews_train_prepared, labels_train_prepared, epochs=10, verbose=False)

# predicting
result = model.predict(reviews_test_prepared)
print(classification_report(labels_test_prepared.argmax(axis=1), result.argmax(axis=1)))

loss, accuracy, f1_score, precision, recall = model.evaluate(reviews_train_prepared, reviews_test_prepared,
                                                             batch_size=batch_size, verbose=1)

print("Accuracy - ", accuracy)
print("F-Measure - ", f1_score)
print("Recall - ", recall)
print("Precision - ", precision)
