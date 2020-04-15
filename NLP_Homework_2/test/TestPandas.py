import pandas

TITLE = 'title'
TEXT = 'text'
LABEL = 'label'

reviews = pandas.read_csv('all_reviews.csv', delimiter=',')
#print(reviews['title'])
print(reviews.iloc[0])