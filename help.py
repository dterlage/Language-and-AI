import spacy
import pandas as pd

data = pd.read_csv("download/extrovert_introvert.csv")

nlp = spacy.blank("en")
#nlp = spacy.load("en_core_web_sm")

corpus_0 = []
corpus_1 = []
for index, row in data.iterrows():
    if row.extrovert == 0:
        corpus_0.append(row.post)
    else:
        corpus_1.append(row.post)
        
tokens_0 = []
tokens_1 = []

for post in nlp.pipe(corpus_0, batch_size=1000):
    tokens_0.extend(token.lower_ for token in post)
for post in nlp.pipe(corpus_1, batch_size=1000):
    tokens_1.extend(token.lower_ for token in post)

tokens_both = set(tokens_0) & set(tokens_1)
only_0 = set(tokens_0) - set(tokens_1)
only_1 = set(tokens_1) - set(tokens_0)

print("Shared tokens:", len(tokens_both))
print("Only in class 0:", len(only_0))
print("Only in class 1:", len(only_1))
