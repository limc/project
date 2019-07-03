import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras import layers
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()


df = pd.read_csv('/Users/clavance/Desktop/Dropbox/Individual_project/pip/singleclass_data.csv', header='infer', encoding='latin1')
df['totalwords'] = df['Text'].str.split().str.len()
# print(df['totalwords'].max())
# print(df['totalwords'].mean())

x = df['Text'].values
# get_dummies changes shape of Classes from 1 to 20 ('pads' others to 0)
y = pd.get_dummies(df['Class']).values

# splits to 80:20 training/test split, stratify means balance dataset split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1000, stratify=y)
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(x_train)

X_train = tokenizer.texts_to_sequences(x_train)
X_test = tokenizer.texts_to_sequences(x_test)
vocab_size = len(tokenizer.word_index) + 1
# maxlen = 470420
maxlen = 2543

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

# because spacy largest model uses 300
embedding_dim = 300
model = Sequential()
model.add(layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen))
model.add(layers.Flatten())
model.add(layers.Dense(10, activation='relu'))
# model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(layers.Dense(20, activation='softmax')) #20 classes
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print(model.summary())

history = model.fit(X_train, y_train,
                    epochs=20,
                    verbose=False,
                    validation_data=(X_test, y_test),
                    batch_size=16)
loss, accuracy = model.evaluate(X_train, y_train, verbose=2)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=2)
print("Testing Accuracy:  {:.4f}".format(accuracy))

plot_history(history)

