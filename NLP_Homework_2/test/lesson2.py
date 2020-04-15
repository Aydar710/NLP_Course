import nltk as nltk
import pymorphy2
import pandas

def extract_pymorphy_tags():
    morph = pymorphy2.MorphAnalyzer()
    p = morph.parse('косой')[0]
    print(p)
    print(p.normal_form)

def read_txt_file():
    f = open("test.txt", encoding="utf-8")
    for line in f:
        words = nltk.word_tokenize(line)
        for word in words:
            print(word)

def read_excel_file():
    df = pandas.read_excel("reviews.xlsx")
    df.columns = ["empty", "text", "rating", "name"]
    df = df["text"]
    for row in df:
        print(row)
# print(len(df.rows()))


