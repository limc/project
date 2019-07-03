import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score

# the text is NOT preprocessed
df = pd.read_csv('/Users/clavance/Desktop/Dropbox/Individual_project/pip/singleclass_data.csv', header='infer', encoding='latin1')
x = df['Text'].values
y = df['Class'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1000, stratify=y)

# vectorizer = CountVectorizer()
# vectorizer.fit(x_train)
vectorizer = TfidfVectorizer(max_features=5000)
vectorizer.fit(x_train)

X_train = vectorizer.transform(x_train)
X_test = vectorizer.transform(x_test)

LR = LogisticRegression()
LR.fit(X_train, y_train)
predictions_LR = LR.predict(X_test)
print("LR Accuracy: ", accuracy_score(predictions_LR, y_test)*100)
# LR Accuracy (with TFIDF vectorizer):  89.95052566481138
# LR Accuracy (with count vectorizer):  93.07359307359307

# SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
# SVM.fit(X_train,y_train)
# predictions_SVM = SVM.predict(X_test)
# print("SVM Accuracy: ", accuracy_score(predictions_SVM, y_test)*100)
# SVM Accuracy (with TFIDF vectorizer): 93.66110080395794
# SVM Accuracy (with count vectorizer): 91.77489177489177

# Naive = naive_bayes.MultinomialNB()
# Naive.fit(X_train,y_train)
# predictions_NB = Naive.predict(X_test)
# print("NB Accuracy: ", accuracy_score(predictions_NB, y_test)*100)
# NB Accuracy (with TFIDF vectorizer): 75.60296846011131
# NB Accuracy (with count vectorizer): 82.40568954854669

