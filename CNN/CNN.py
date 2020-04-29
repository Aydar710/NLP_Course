import keras
import pandas
from keras.layers import Dense, Activation, Embedding, Conv1D, GlobalMaxPool1D
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import classification_report
from Utils import deserialize_normalized_reviews, deserialize_my_normalized_reviews, create_embedding_matrix, \
    serialize_ruscorpora_data

max_len = 300

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

# Use deserialized data further
i = 0
for row in ruscorpora_data["text"]:
    ruscorpora_data["text"][i] = row.split("_", 1)[0]
    i += 1
i = 0
for row in ruscorpora_data["value"]:
    ruscorpora_data["value"][i] = row.split(" ")
    i += 1

serialize_ruscorpora_data(ruscorpora_data)

tokenizer = Tokenizer(num_words=ruscorpora_data.size)
tokenizer.fit_on_texts(ruscorpora_data["text"])

# making reviews as sequences
reviews_to_test = tokenizer.texts_to_sequences(reviews_to_test)
reviews_to_train = tokenizer.texts_to_sequences(reviews_to_train)

# preparing reviews
reviews_train_prepared = pad_sequences(reviews_to_test.to_numpy(), maxlen=max_len, padding='post')
reviews_test_prepared = pad_sequences(reviews_to_train.to_numpy(), maxlen=max_len, padding='post')

# get categorical
labels_train_prepared = keras.utils.to_categorical(test_labels, 3)
labels_test_prepared = keras.utils.to_categorical(all_reviews["label"], 3)

# create emdedding matrix
embedding_matrix = create_embedding_matrix(ruscorpora_data, tokenizer.word_index, max_len)

input_length = 300

# Building neural network
model = Sequential()
model.add(Embedding(ruscorpora_data.size, input_length, weights=[embedding_matrix], input_length=input_length, trainable=False))
model.add(Conv1D(input_length, 3))
model.add(Activation("relu"))
model.add(GlobalMaxPool1D())
model.add(Dense(9))
model.add(Activation('softmax'))
model.compile(metrics=["accuracy"], optimizer='adam', loss='binary_crossentropy')

# fitting model
#model.fit(reviews_train_prepared, labels_train_prepared, epochs=30, verbose=False)
# model.fit(reviews_train_prepared, labels_train_prepared, epochs=20, verbose=False)
model.fit(reviews_train_prepared, labels_train_prepared, epochs=10, verbose=False)

# predicting
result = model.predict(reviews_test_prepared)
print(classification_report(labels_test_prepared.argmax(axis=1), result.argmax(axis=1)))