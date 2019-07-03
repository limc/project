import gensim
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec as w2v
from gensim.models.phrases import Phrases, Phraser
import warnings
warnings.filterwarnings(action='ignore')
import os
from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix


base_dir = "/Users/clavance/Desktop/Dropbox/Individual_project/EURLEX/html_tokenised_lemmatised/"
directory = os.fsencode(base_dir)

#initialise an empty model
# min_count: ignore words with lower frequency than the count
# window: maximum distance between the current and predicted word within a sentence
# size: dimensionality of the feature vectors
# alpha: learning rate
model = w2v(min_count=10, window=2, sample=6e-5, negative=20, alpha=0.03, min_alpha=0.0007, size=300)

#initialise empty list of dictionaries to create pandas dataframe
items = []

for file in os.listdir(directory):
    dict = {}
    filename = os.fsdecode(file)
    id = filename.split(".txt", 1)[0]
    dict["ID"] = id

    #text is already tokenised using lexnlp
    r = open(base_dir+filename, "r", encoding='latin1').read()
    s = r.split("Class: ",1)[1]
    classes = s.split("\nText: ", 1)[0]
    dict["Class"] = classes
    text = s.split("\nText: ", 1)[1]
    dict["Text"] = text
    items.append(dict)

    # model.build_vocab(text)

#create pandas dataframe from data
df = pd.DataFrame(items)

#use gensim phrases package to detect common phrases
#split each text item into words
sent = [row.split() for row in df["Text"]]

#Phrases takes in as argument a list of words
#ignore all bigrams with collected count lower than 30
phrases = Phrases(sent, min_count=30)

#standard gensim syntax
#finds bigrams and stores them as one token
bigram = Phraser(phrases)
sentences = bigram[sent]
# print(list(sentences))

t = time()
model.build_vocab(sentences, progress_per=1000)
print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

u = time()
model.train(sentences, total_examples=model.corpus_count, epochs=30, report_delay=1)
print('Time to train the model: {} mins'.format(round((time() - u) / 60, 2)))

model.save("word2vec_second.model")

#using seaborn, plot the results from the t-SNE dimensionality reduction algorithm
# of the vectors of a query word, its list of most similar words, and a list of words.
# this function is from pierremegret/gensim-word2vec-tutorial
def tsnescatterplot(w2vmodel, word, list_names, savename):
    arrays = np.empty((0,300), dtype='f')
    word_labels = [word]
    color_list = ['red']
    arrays = np.append(arrays, w2vmodel.wv.__getitem__([word]), axis=0)
    close_words = w2vmodel.wv.most_similar([word])

    for wrd_score in close_words:
        wrd_vector = w2vmodel.wv.__getitem__([wrd_score[0]])
        word_labels.append(wrd_score[0])
        color_list.append('blue')
        arrays = np.append(arrays, wrd_vector, axis=0)

    for wrd in list_names:
        wrd_vector = w2vmodel.wv.__getitem__([wrd])
        word_labels.append(wrd)
        color_list.append('green')
        arrays = np.append(arrays, wrd_vector, axis=0)

    reduc = PCA(n_components=50).fit_transform(arrays)
    np.set_printoptions(suppress=True)
    Y = TSNE(n_components=2, random_state=0, perplexity=15).fit_transform(reduc)

    df1 = pd.DataFrame({'x': [x for x in Y[:, 0]],
                       'y': [y for y in Y[:, 1]],
                       'words': word_labels,
                       'color': color_list})

    fig, _ = plt.subplots()
    fig.set_size_inches(9, 9)

    p1 = sns.regplot(data=df1,
                     x="x",
                     y="y",
                     fit_reg=False,
                     marker="o",
                     scatter_kws={'s': 40,
                                  'facecolors': df1['color']
                                 }
                    )

    for line in range(0, df1.shape[0]):
        p1.text(df1["x"][line],
                df1['y'][line],
                '  ' + df1["words"][line].title(),
                horizontalalignment='left',
                verticalalignment='bottom', size='medium',
                color=df1['color'][line],
                weight='normal'
                ).set_size(15)

    plt.xlim(Y[:, 0].min() - 50, Y[:, 0].max() + 50)
    plt.ylim(Y[:, 1].min() - 50, Y[:, 1].max() + 50)

    plt.title('t-SNE visualization for {}'.format(word.title()))

    plt.savefig(savename+'.png')

#compare vector representation of 10 most similar words to "agreement", and 8 more random words, in a graph
tsnescatterplot(model, 'agreement', ['cooperation', 'parties', 'international', 'article', 'european', 'committee', 'organization', 'information'],'agreement_random_lemmatised_2')

tsnescatterplot(model, 'agreement', [i[0] for i in model.wv.most_similar(negative=["agreement"])], 'agreement_negative_lemmatised_2')
