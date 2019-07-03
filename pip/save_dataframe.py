import pandas as pd
import os

base_dir = "/Users/clavance/Desktop/Dropbox/Individual_project/EURLEX/html_clean_single_classed/"
directory = os.fsencode(base_dir)

items=[]
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    r = open(base_dir+filename, "r", encoding='latin1').read()
    s = r.split("Class: ",1)[1]
    classes = s.split("\nText: \n", 1)[0]
    text = s.split("\nText: \n", 1)[1]
    dict = {}
    dict["Class"] = classes
    dict["Text"] = text
    items.append(dict)

df = pd.DataFrame(items)
df.to_csv('/Users/clavance/Desktop/Dropbox/Individual_project/pip/singleclass_data.csv',index=False,header=True,encoding='latin1')
