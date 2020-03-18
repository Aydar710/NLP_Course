import pickle

import nltk
import pandas
import pymorphy2
from collections import defaultdict
from collections import OrderedDict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support

analyzer = pymorphy2.MorphAnalyzer()
my_reviews_df = pandas.read_csv('reviewcsvs/my_reviews.csv', delimiter=',')


def serialize_normalized_words(normallized_words):
    pickle_out = open("all_normallized_words.pickle", "wb")
    pickle.dump(normallized_words, pickle_out)
    pickle_out.close()


def deserialize_normalized_words():
    pickle_in = open("all_normallized_words.pickle", "rb")
    normalized_docs = pickle.load(pickle_in)
    return normalized_docs


def deserialize_normalized_reviews():
    pickle_in = open("normallized_list.pickle", "rb")
    normalized_reviews = pickle.load(pickle_in)
    return normalized_reviews


def serialize_bag_of_words(bag_of_words):
    pickle_out = open("bag_of_words.pickle", "wb")
    pickle.dump(bag_of_words, pickle_out)
    pickle_out.close()


def deserialize_bag_of_words():
    pickle_in = open("bag_of_words.pickle", "rb")
    return pickle.load(pickle_in)


def deserialize_my_reviews_normalized():
    pickle_in = open("my_reviews_normalized.pickle", "rb")
    return pickle.load(pickle_in)


def normalize_words(rows):
    result = []
    i = 0
    for row in rows.itertuples():
        words = nltk.word_tokenize(row.text)
        for word in words:
            word = analyzer.parse(word)[0]
            result.append(word.normal_form)
            i += 1
            print(i)
    return result


def get_bag_of_words(normalized_words, all_reviews):
    bag = []
    i = 0
    for review in all_reviews:
        frequencies_from_review = []
        for word in normalized_words:
            frequencies = review.count(word)
            frequencies_from_review.append(frequencies)
        bag.append(frequencies_from_review)
        i += 1
        print(i)
    return bag


reviews = pandas.read_csv('reviewcsvs/reviews.csv', delimiter=',')

# norm_words = normalize_words(reviews)
# serialize_normalized_words(norm_words)
review_labels = reviews["label"].to_numpy()
all_normalized_words = deserialize_normalized_words()
normalized_reviews = deserialize_normalized_reviews()
normalized_words_set = set()

for normalized_words in all_normalized_words:
    for word in normalized_words:
        normalized_words_set.add(word)

# bag_of_words = get_bag_of_words(normalized_words_set, normalized_reviews)

# serialize_bag_of_words(bag_of_words)

bag_of_words = deserialize_bag_of_words()

# Training
print("Running Training Model")
model = LogisticRegression(max_iter=10000)
model.fit(bag_of_words, review_labels)
print("Model is ready")

bag_of_word_my_reviews = get_bag_of_words(normalized_words_set, deserialize_my_reviews_normalized())
predictions = model.predict(bag_of_word_my_reviews)
print("")

# Calculate accuracy
true_positives = 0
k = 0
for prediction in predictions:
    if prediction == my_reviews_df.values[k][2]:
        true_positives += 1
    k += 1

accuracy = true_positives / len(predictions)
print("Accuracy: " + str(accuracy))

# Calculate recall, precision, f-measure
precision_recall_fscore = precision_recall_fscore_support(my_reviews_df['label'].values, predictions)
print("precision " + str(precision_recall_fscore[0]))
print("recall " + str(precision_recall_fscore[1]))
print("f-measure " + str(precision_recall_fscore[2]))

# get weights
negative_dict = dict(zip(normalized_words_set, model.coef_[0]))
neutral_dict = dict(zip(normalized_words_set, model.coef_[1]))
positive_dict = dict(zip(normalized_words_set, model.coef_[2]))

# sort dicts
sorted_negative_dictionary = {k: v for k, v in sorted(negative_dict.items(), key=lambda item: item[1], reverse=True)}
sorted_neutral_dictionary = {k: v for k, v in sorted(neutral_dict.items(), key=lambda item: item[1], reverse=True)}
sorted_positive_dictionary = {k: v for k, v in sorted(positive_dict.items(), key=lambda item: item[1], reverse=True)}

# mapping negative
sorted_negative_list = list(sorted_negative_dictionary)
first_negative = sorted_negative_list[0:10]
print("Top negative")
print(first_negative)
reversed_sorted_negative_list = sorted_negative_list[::-1]
last_negative = reversed_sorted_negative_list[0:10]
print("Last negative")
print(last_negative)

# mapping positives
sorted_positive_list = list(sorted_positive_dictionary)
first_positives = sorted_positive_list[0:10]
print("Top positives:")
print(first_positives)
descending_positive = sorted_positive_list[::-1]
last_positives = descending_positive[0:10]
print("Last positives")
print(last_positives)

# mapping neutral
sorted_neutral_list = list(sorted_neutral_dictionary)
first_neutral = sorted_neutral_list[0:10]
print("Top Neutral")
print(first_neutral)
descending_neutral = sorted_neutral_list[::-1]
last_10_neutral = descending_neutral[0:10]
print("Last Neutral")
print(last_10_neutral)
