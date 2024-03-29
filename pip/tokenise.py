import pandas as pd
import os
import lexnlp.nlp.en.tokens as ln
from sklearn.feature_extraction.text import TfidfVectorizer as tfv

df = pd.read_csv("/Users/clavance/Desktop/Dropbox/Individual_project/code/classifications.csv", delimiter=' ', header='infer')

dir = "/Users/clavance/Desktop/Dropbox/Individual_project/EURLEX/html_clean/"
second_dir = "/Users/clavance/Desktop/Dropbox/Individual_project/EURLEX/html_tokenised/"
directory = os.fsencode(dir)

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    celex = filename.split(".txt", 1)[0]
    print(celex)
    f = open(dir+filename, "r", encoding='latin1').read()
    h = f.split("\nTitle: ", 1)
    title = h[1].split("\nText: ", 1)[0]
    print(title)
    text = h[1].split("\nText: ", 1)[1]
    tokens = ln.get_token_list(text, stopword=True)
    tokenstring = " ".join(tokens)

    for i in range(len(df)):
        if df.loc[i, 'CelexID'] == celex:

            if df.loc[i+3, 'CelexID'] == celex:
                classification = df.loc[i, 'Classes']
                classification2 = df.loc[i+1, 'Classes']
                classification3 = df.loc[i+2, 'Classes']
                classification4 = df.loc[i+3, 'Classes']
                docid = df.loc[i, 'DocID']

                with open(second_dir + str(docid) + '.txt', "w", encoding='latin1') as newfile:
                    newfile.write("Class: " + str(classification) + "," + str(classification2) + "," +
                                  str(classification3) + "," + str(classification4) + "\n" + "Text: " + tokenstring)

            if df.loc[i+2, 'CelexID'] == celex:
                classification = df.loc[i, 'Classes']
                classification2 = df.loc[i+1, 'Classes']
                classification3 = df.loc[i+2, 'Classes']
                docid = df.loc[i, 'DocID']

                if os.path.isfile(second_dir + str(docid) + '.txt'):
                    continue
                else:
                    with open(second_dir + str(docid) + '.txt', "w", encoding='latin1') as newfile:
                        newfile.write("Class: " + str(classification) + "," + str(classification2) + "," +
                                      str(classification3) + "\n" + "Text: " + tokenstring)

            elif df.loc[i+1, 'CelexID'] == celex:
                classification = df.loc[i, 'Classes']
                classification2 = df.loc[i+1, 'Classes']
                docid = df.loc[i, 'DocID']

                if os.path.isfile(second_dir + str(docid) + '.txt'):
                    continue
                else:
                    with open(second_dir + str(docid) + '.txt', "w", encoding='latin1') as newfile:
                        newfile.write("Class: " + str(classification) + "," + str(classification2) + "\n" + "Text: " + tokenstring)

            else:
                classification = df.loc[i, 'Classes']
                docid = df.loc[i, 'DocID']

                if os.path.isfile(second_dir + str(docid) + '.txt'):
                    continue
                else:
                    with open(second_dir + str(docid) + '.txt', "w", encoding='latin1') as newfile:
                        newfile.write("Class: " + str(classification) + "\n" + "Text: " + tokenstring)
