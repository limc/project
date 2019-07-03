import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('/Users/clavance/Desktop/Dropbox/Individual_project/pip/singleclass_data.csv', header='infer', encoding='latin1')
x = df['Text'].values
y = df['Class'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1000, stratify=y)
vectorizer = CountVectorizer()
vectorizer.fit(x_train)

X_train = vectorizer.transform(x_train)
X_test = vectorizer.transform(x_test)

classifier = LogisticRegression()
classifier.fit(X_train, y_train)
score = classifier.score(X_test, y_test)

print("Accuracy:", score)
# without stratify:
# Accuracy: 0.9310451453308596
# with stratify:
# Accuracy: 0.9307359307359307


