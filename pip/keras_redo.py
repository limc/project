import pandas as pd
import string
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.model_selection import train_test_split

nlp = spacy.load('en_core_web_lg')
nlp.max_length = 2871868
punctuations = string.punctuation
stopwords = spacy.lang.en.stop_words.STOP_WORDS

df = pd.read_csv('/Users/clavance/Desktop/Dropbox/Individual_project/pip/singleclass_data.csv', header='infer', encoding='latin1')

def cleanup_text(docs, logging=False):
    texts = []
    counter = 1
    for doc in docs:
        if counter % 1000 == 0 and logging:
            print("Processed %d out of %d documents." % (counter, len(docs)))
        counter += 1
        doc = nlp(doc, disable=['parser', 'ner'])
        tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
        tokens = [tok for tok in tokens if tok not in stopwords and tok not in punctuations]
        tokens = ' '.join(tokens)
        texts.append(tokens)
    return pd.Series(texts)

# process raw text
print('Original training data shape: ', df['Text'].shape)
train_cleaned = cleanup_text(df['Text'], logging=True)
print('Cleaned up training data shape: ', train_cleaned.shape)

#vectorise?
x = [doc.vector for doc in nlp.pipe(train_cleaned, batch_size=500)]
print('Total number of documents parsed: {}'.format(len(x)))
print('Size of vector embeddings: ', x.shape[1])
print('Shape of vectors embeddings matrix: ', x.shape)

# get_dummies changes shape of labels from 1 to 20 ('pads' others to 0)
y = pd.get_dummies(df['Class']).values
print('Shape of class embeddings: ', y.shape)

#split dataset into training:test 80:20
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

# keras model
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras import layers #?

model = Sequential()
model.add(layers.Dense(128, activation='relu', input_dim=300))
model.add(Dropout(0.2))
# model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(layers.Dense(20, activation='softmax')) # for 20 classes
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
print(model.summary())

estimator = model.fit(X_train, y_train,
                    epochs=50,
                    validation_split=0.20,
                    # validation_data=(X_test, y_test),
                    batch_size=128,
                    verbose=1)

print("Training accuracy: %.2f%% / Validation accuracy: %.2f%%" %
      (100*estimator.history['acc'][-1], 100*estimator.history['val_acc'][-1]))

import matplotlib.pyplot as plt

# model accuracy over epochs
plt.plot(estimator.history['acc'])
plt.plot(estimator.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()
plt.savefig('model_acc.png')

# model loss over epochs
plt.plot(estimator.history['loss'])
plt.plot(estimator.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()
plt.savefig('model_loss.png')

loss, accuracy = model.evaluate(X_train, y_train, verbose=1)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print("Testing Accuracy:  {:.4f}".format(accuracy))