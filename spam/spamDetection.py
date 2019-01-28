import csv
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import  Pipeline
from sklearn.linear_model import SGDClassifier
import re
from nltk.stem.snowball import EnglishStemmer
from nltk.corpus import  stopwords
import numpy as np

def loadData(file):
    messages = csv.reader(open(file, encoding = "utf-8"))
    data = []
    y = []
    for r in messages:
        if r[0] == "Label":
            continue
        elif r[0] == "ham":
            y.append(0)
        elif r[0] == "spam":
            y.append(1)
        data.append(r[1])
    return data, y

def clean_text(comment_text):
    stop_words = stopwords.words('english')
    porter = EnglishStemmer()
    comment_list = []
    for text in comment_text:
        text = text.lower()
        text = re.sub(r'\b[\w\-.]+?@\w+?\.\w{2,4}\b', 'emailaddr', text)
        text = re.sub(r'(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)', 'httpaddr', text)
        text = re.sub(r'£|\$', 'moneysymb', text)
        text = re.sub(r'\b(\+\d{1,2}\s)?\d?[\-(.]?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b', 'phonenumbr', text)
        text = re.sub(r'\d+(\.\d+)?', 'numbr', text)
        text = re.sub(r'[^\w\d\s]', ' ', text) #remove punctuation
        text = re.sub(r'\s+', ' ', text) #collapse all whitespace into a single space
        text = re.sub(r'^\s+|\s+?$', '', text) #remove any leading or trailing whitespace

        #stemming on non-stop words
        new_text =  ' '.join(
            porter.stem(term)
            for term in text.split()
            if term not in set(stop_words)
        )
        comment_list.append(new_text)
    return comment_list

train_data, train_y = loadData("train.csv")
train_data = clean_text(train_data)
text_clf = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1, 2))),
    ('tfidf', TfidfTransformer(use_idf=True, smooth_idf=True)),
    ('clf', SGDClassifier(loss='hinge')),
])
text_clf.fit(train_data, train_y)

test_data, test_y = loadData("test.csv")
test_data = clean_text(test_data)
predicted = text_clf.predict(test_data)
print(np.mean(test_y == predicted))
