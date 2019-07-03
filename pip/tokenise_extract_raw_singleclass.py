import os
from shutil import copy2

dir = "/Users/clavance/Desktop/Dropbox/Individual_project/EURLEX/html_clean_multi_classed/"
second_dir = "/Users/clavance/Desktop/Dropbox/Individual_project/EURLEX/html_clean_single_classed/"

directory = os.fsencode(dir)

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    f = open(dir+filename, "r", encoding='latin1').read()

    h = f.split("Class: ", 1)
    classes = h[1].split("\nText: \n", 1)[0]

    if ',' in classes:
        continue
    else:
        copy2(dir+filename, second_dir)