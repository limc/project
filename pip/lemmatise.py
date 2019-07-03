import os
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

source_dir = "/Users/clavance/Desktop/Dropbox/Individual_project/EURLEX/html_tokenised/"
target_dir = "/Users/clavance/Desktop/Dropbox/Individual_project/EURLEX/html_tokenised_lemmatised/"

directory = os.fsencode(source_dir)

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    docid = filename.split(".txt", 1)[0]
    r = open(source_dir+filename, "r", encoding='latin1').read()
    s = r.split("Class: ",1)[1]
    classes = s.split("\nText: `` ", 1)[0]
    text = s.split("\nText: `` ", 1)[1]
    # remove quotation marks at the end of text
    text = text[:-3]

    # make text lowercase
    text_str = text.lower()
    # remove numbers
    text_num = re.sub(r"\d+", "", text_str)
    # remove punctuation
    text_punc = re.sub(r'[^\w\s]','', text_num)
    # remove trailing whitespaces
    text_strip = text_punc.strip()
    # remove double whitespaces
    text_space = text_strip.replace("  ", " ")
    # lemmatise (but NOT stemmed)
    lemmatizer = WordNetLemmatizer()
    # lemmatisation returns a list of words
    # (consider what you want the output to be?)
    text_lem = word_tokenize(text_space)
    # output as string of words
    text_string = " ".join(text_lem)
    print(text_string)

    with open(target_dir + str(docid) + '.txt', "w", encoding='latin1') as f:
        f.write("Class: " + str(classes) + "\n" + "Text: " + text_string)