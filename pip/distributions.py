import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import time
import seaborn as sns
import matplotlib.pyplot as plt

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
#converts Class values from string to int
df['Class'] = df['Class'].astype(int)
#prints the distribution of classes
# print(df['Class'].value_counts().sort_index())

#plots bar chart of the distribution of classes
x_list = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20']
class_dist = sns.barplot(x=x_list, y=df['Class'].value_counts().sort_index(), order=x_list)
class_dist.set_xlabel('Class')
class_dist.set_ylabel('Number of Samples')
plt.show()

# t0 = time.time()
