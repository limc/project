import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import time

base_dir = "/Users/clavance/Desktop/Dropbox/Individual_project/EURLEX/html_clean_single_classed/"
directory = os.fsencode(base_dir)

items = []

for file in os.listdir(directory):
    dict = {}
    filename = os.fsdecode(file)
    docid = filename.split(".txt", 1)[0]
    dict["ID"] = docid

    r = open(base_dir+filename, "r", encoding='latin1').read()
    s = r.split("Class: ",1)[1]
    classification = s.split("\nText: \n", 1)[0]
    dict["Class"] = classification
    text = s.split("\nText: \n", 1)[1]
    dict["Text"] = text

    items.append(dict)

df = pd.DataFrame(items)
sentences = df["Text"].values
y = df["Class"].values

t0 = time.time()

#split into training (80%) and test sets (20%)
sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.2, random_state=42)
vectorizer = CountVectorizer()
vectorizer.fit(sentences_train)

X_train = vectorizer.transform(sentences_train)
X_test = vectorizer.transform(sentences_test)

clf = LogisticRegression(class_weight='balanced', multi_class='multinomial', solver='lbfgs')
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
print("Test score: %.4f" % score)

run_time = time.time() - t0
print('Example run in %.3f s' % run_time)

#Test score: 0.8958
#Example run in 421.865 s
