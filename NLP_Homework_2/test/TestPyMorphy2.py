import pymorphy2
import nltk

morph = pymorphy2.MorphAnalyzer()

sentence = nltk.word_tokenize("Идущий человек")
for word in sentence:
    parsedWord = morph.parse(word)[0]
    print(parsedWord.normal_form)