import pickle

import pandas
import nltk
import pymorphy2

reviews_df = pandas.read_csv('reviewcsvs/reviews.csv', delimiter=',')
my_reviews_df = pandas.read_csv('reviewcsvs/my_reviews.csv', delimiter=',')

morph = pymorphy2.MorphAnalyzer()


def normalize(reviews):
    normallized_docs = []
    for i in range(reviews.__len__()):
        normallized_text = []
        for word in nltk.word_tokenize(reviews.at[i, 'text']):
            normallized_text.append(morph.parse(word)[0].normal_form)
        reviews.at[i, 'text'] = normallized_text
        normallized_docs.append(normallized_text)
        print(i)
    return normallized_docs


# normalized_docs = normalize(reviews_df)
# pickle_out = open("normallized_list.pickle", "wb")
# pickle.dump(normalized_docs, pickle_out)
# pickle_out.close()

my_reviews_normalized = normalize(my_reviews_df)
pickle_out = open("my_reviews_normalized.pickle", "wb")
pickle.dump(my_reviews_normalized, pickle_out)
pickle_out.close()
