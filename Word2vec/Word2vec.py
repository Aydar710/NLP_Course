import gensim.models
import pandas
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support

from Utils import deserialize_normalized_reviews, serialize_model, deserialize_model, get_average_feature_vecs, \
    deserialize_my_normalized_reviews

reviews_to_train = deserialize_normalized_reviews()
all_reviews = pandas.read_csv('data/normallized_reviews.csv', delimiter=',')

# Creating word2vec model
model = gensim.models.Word2Vec(reviews_to_train, min_count=2, iter=200)

# Serializing model
# serialize_model(model)
# model = deserialize_model()

# Printing synonyms
print(model.wv.most_similar(positive=['разум'], topn=3))
print(model.wv.most_similar(positive=['творчество'], topn=3))
print(model.wv.most_similar(positive=['доброта'], topn=3))
print(model.wv.most_similar(positive=['чувство'], topn=3))
print(model.wv.most_similar(positive=['мрак'], topn=3))

# get testing reviews
reviews_to_test = deserialize_my_normalized_reviews()
test_labels = pandas.read_csv('data/my_labels.csv', delimiter=',')

# Getting train and test vecs
train_vecs = get_average_feature_vecs(reviews_to_train, model, 100)
test_vecs = get_average_feature_vecs(reviews_to_test, model, 100)

# Training model using RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=0)
classifier.fit(train_vecs, all_reviews['label'])

# Get predictions on test_vecs
predictions = classifier.predict(test_vecs)

# Calculating accuracy
correct_prediction_count = 0
i = 0
for prediction in predictions:
    if prediction == all_reviews.values[i][3]:
        correct_prediction_count += 1
    i += 1

accuracy = correct_prediction_count / len(predictions)
metrics = precision_recall_fscore_support(test_labels, predictions)

print("Accuracy: {}".format(accuracy))
print(f"Precision(-1, 0, 1) = {metrics[0]}")
print(f"Recall(-1, 0, 1) = {metrics[1]}")
print(f"F-Score(-1, 0, 1) = {metrics[2]}")
