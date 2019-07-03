import lexnlp.nlp.en.tokens as ln

dir = "/Users/clavance/Desktop/Dropbox/Individual_project/EURLEX/html_clean/42006D0281.txt"
celex = "42006X0325(01)"

f = open(dir, "r", encoding='latin1').read()
h = f.split("\nTitle: ", 1)
title = h[1].split("\nText: ", 1)[0]
print(title)
text = h[1].split("\nText: ", 1)[1]
tokens = ln.get_token_list(text, stopword=True)
tokenstring = " ".join(tokens)

print(tokenstring)
