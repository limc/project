import os
import time
import spacy
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
import string
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
from spacy import displacy

t0 = time.time()

#load spaCy largest model
nlp = spacy.load('en_core_web_sm')
stop_words = spacy.lang.en.stop_words.STOP_WORDS
punctuations = string.punctuation

base_dir = "/Users/clavance/Desktop/Dropbox/Individual_project/EURLEX/html_clean_single_classed/"
directory = os.fsencode(base_dir)

df = pd.read_csv('/Users/clavance/Desktop/Dropbox/Individual_project/pip/singleclass_data.csv', header='infer', encoding='latin1')
X = df['Text']
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
classifier = LogisticRegression()

# Basic function to clean the text
def clean_text(text):
    # Removing spaces and converting text into lowercase
    return text.strip().lower()

# create tokenizer function using spacy
def spacy_tokenizer(data):
    tokens = []
    doc = nlp(data)

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

    return tokens

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    r = open(base_dir+filename, "r", encoding='latin1').read()
    s = r.split("Class: ",1)[1]
    classes = s.split("\nText: \n", 1)[0]
    text = s.split("\nText: \n", 1)[1]

    tokens = spacy_tokenizer(text)

    tokens_string = " ".join(tokens)
    doc2 = nlp(tokens_string)

    # bag of words and unigrams
    bow_vector = CountVectorizer(tokenizer=spacy_tokenizer, ngram_range=(1,1))
    tfidf_vector = TfidfVectorizer(tokenizer=spacy_tokenizer)

    # load processed tokens to df


    # #get tokenised list from doc object
    # #this is word tokenisation (cf. sentence tokenisation)
    # token_list = []
    # for token in doc:
    #     token_list.append(token.text)
    #
    # text_nostopword = []
    # for word in doc:
    #     if word.is_stop==False:
    #         text_nostopword.append(word)
    # print(text_nostopword)

    break




    # #load each sample into spaCy model
    # #loading results in a doc object
    # doc = nlp(text)
    #
    # # once loaded into the model, each doc item already has a vectorised form
    # # print(doc.vector.shape)
    # # print(doc.vector)
    #
    # # tokens = list of tokens from text
    # # tokens = list(word.text for word in doc)
    # # print(tokens)
    #
    # # lemmatise and convert to lowercase if not pronoun
    # tokens = []
    # for word in doc:
    #     if (word.lemma_ != "-PRON-"):
    #         tokens.append(word.lemma_.lower().strip())
    #     else:
    #         tokens.append(word.lower_)
    #
    # # remove stopwords, punctutation
    # for word in tokens:
    #     if word in stop_words or word in punctuations:
    #         tokens.remove(word)