import pandas as pd
import string
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras import layers
import matplotlib.pyplot as plt


# def plot_history(history):
#     acc = history.history['acc']
#     val_acc = history.history['val_acc']
#     loss = history.history['loss']
#     val_loss = history.history['val_loss']
#     x = range(1, len(acc) + 1)
#
#     plt.figure(figsize=(12, 5))
#     plt.subplot(1, 2, 1)
#     plt.plot(x, acc, 'b', label='Training acc')
#     plt.plot(x, val_acc, 'r', label='Validation acc')
#     plt.title('Training and validation accuracy')
#     plt.legend()
#     plt.subplot(1, 2, 2)
#     plt.plot(x, loss, 'b', label='Training loss')
#     plt.plot(x, val_loss, 'r', label='Validation loss')
#     plt.title('Training and validation loss')
#     plt.legend()


def spacy_tokenizer(data):
    tokens = []
    doc = nlp(data, disable=['parser', 'ner'])

    # lemmatise and convert to lowercase if not pronoun
    for word in doc:
        if (word.lemma_ != "-PRON-"):
            tokens.append(word.lemma_.lower().strip())
        else:
            tokens.append(word.lower_)

    # remove stopwords, punctutation
    for word in tokens:
        if word in stop_words or word in punctuations:
            tokens.remove(word)

    tokens_string = " ".join(tokens)
    return tokens_string

stop_words = spacy.lang.en.stop_words.STOP_WORDS
punctuations = string.punctuation

def cleanup_text(docs, logging=False):
    texts = []
    counter = 1
    for doc in docs:
        if counter % 1000 == 0 and logging:
            print("Processed %d out of %d documents." % (counter, len(docs)))
        counter += 1
        doc = nlp(doc, disable=['parser', 'ner'])
        # lowercase if not pronoun
        tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
        # remove stopwords, punctuation
        tokens = [tok for tok in tokens if tok not in stop_words and tok not in punctuations]
        tokens = ' '.join(tokens)
        texts.append(tokens)
    return pd.Series(texts)

nlp = spacy.load('en_core_web_lg')
nlp.max_length = 2871868

df = pd.read_csv('/Users/clavance/Desktop/Dropbox/Individual_project/pip/singleclass_data.csv', header='infer', encoding='latin1')

# store all text as a list
all_text = [text for text in df['Text']]

# pass list of all text to cleanup function
# text_clean = cleanup_text(all_text)
# text_clean = ' '.join(text_clean).split()
# text_clean = [word for word in text_clean if word != '\'s']

# df['Preprocessed'] = df['Text'].apply(spacy_tokenizer)

x = df['Preprocessed'].values
# get_dummies changes shape of Classes from 1 to 20 ('pads' others to 0)
y = pd.get_dummies(df['Class']).values
# # splits to 80:20 training/test split, stratify means balance dataset split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1000, stratify=y)

X_train = nlp(x_train)
X_test = nlp(x_test)

# df['totalwords'] = df['Text'].str.split().str.len()
# print(df['totalwords'].max())
# print(df['totalwords'].mean())

# x = df['Preprocessed'].values
# # get_dummies changes shape of Classes from 1 to 20 ('pads' others to 0)
# y = pd.get_dummies(df['Class']).values
#
# # splits to 80:20 training/test split, stratify means balance dataset split
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1000, stratify=y)
#
# X_train = tokenizer.texts_to_sequences(x_train)
# X_test = tokenizer.texts_to_sequences(x_test)
# vocab_size = len(tokenizer.word_index) + 1
# # maxlen = 470420
# maxlen = 2543
#
# X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
# X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
#
# # because spacy largest model uses 300
# embedding_dim = 300
# model = Sequential()
# model.add(layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen))
# model.add(layers.Flatten())
# model.add(layers.Dense(10, activation='relu'))
# # model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
# model.add(layers.Dense(20, activation='softmax')) #20 classes
# model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
#
# print(model.summary())
#
# history = model.fit(X_train, y_train,
#                     epochs=20,
#                     verbose=False,
#                     validation_data=(X_test, y_test),
#                     batch_size=16)
# loss, accuracy = model.evaluate(X_train, y_train, verbose=2)
# print("Training Accuracy: {:.4f}".format(accuracy))
# loss, accuracy = model.evaluate(X_test, y_test, verbose=2)
# print("Testing Accuracy:  {:.4f}".format(accuracy))
#
# plot_history(history)
#
