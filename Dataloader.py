#%% run only once, at the start
#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
#load data
data = pd.read_csv("download/extrovert_introvert.csv")

# %%
# ---------------------- split randomly 80/10/10
train, val_test = train_test_split(data, test_size=0.2, random_state=2026)
val, test = train_test_split(val_test, test_size=0.5, random_state=2026)

#we are predicting extrovert/itnrovert = y
X_train = train.drop('extrovert', axis=1).copy()
y_train = train[['extrovert']].copy()

#validation data
X_val = val.drop('extrovert', axis=1).copy()
y_val = val[['extrovert']].copy()

#testing data
X_test = test.drop('extrovert', axis=1).copy()
y_test = test[['extrovert']].copy()
 # %%
 # ------------------- Stratified split
# split dataset 80/10/10 with all sets having equal density int/ex (stratified split)
# use random state for reproducibility
train_s, val_test_s = train_test_split(data, test_size=0.2, stratify=data.extrovert, random_state=2026)
val_s, test_s = train_test_split(val_test_s, test_size=0.5, stratify=val_test_s.extrovert, random_state=2026)

#we are predicting extrovert/itnrovert = y
X_train_s = train_s.drop('extrovert', axis=1).copy()
y_train_s = train_s[['extrovert']].copy()

#validation data
X_val_s = val_s.drop('extrovert', axis=1).copy()
y_val_s = val_s[['extrovert']].copy()

#testing data
X_test_s = test_s.drop('extrovert', axis=1).copy()
y_test_s = test_s[['extrovert']].copy()
# %%
# Correct amount of extrovert for the majority model
y_true_val = val.extrovert.values
y_true_test = test.extrovert.values

# %%
#this makes two separate lists for posts from extroverts and introverts
corpus_0 = []
corpus_1 = []
for index, row in data.iterrows():
    if row.extrovert == 0:
        corpus_0.append(row.post)
    else:
        corpus_1.append(row.post)

# %%
# Vectorizing the data sets

# Extract text only for ML 
X_train_text = X_train["post"].astype(str)
X_val_text   = X_val["post"].astype(str)
X_test_text  = X_test["post"].astype(str)

# Extract text only for ML
X_train_s_text = X_train_s["post"].astype(str)
X_val_s_text   = X_val_s["post"].astype(str)
X_test_s_text  = X_test_s["post"].astype(str)

# Vecttorizer
vectorizer = TfidfVectorizer(
    lowercase=True,         # Convert all characters to lowercase before tokenizing.
    analyzer="word",
    stop_words="english",   #removes stop words 
    max_features=5000,      
    ngram_range=(1, 2)      
)

# Transforming the text into vectors
X_train_vec = vectorizer.fit_transform(X_train_text)
X_val_vec   = vectorizer.transform(X_val_text)
X_test_vec  = vectorizer.transform(X_test_text)

# Transforming the text into vectors
X_train_s_vec = vectorizer.fit_transform(X_train_s_text)
X_val_s_vec   = vectorizer.transform(X_val_s_text)
X_test_s_vec  = vectorizer.transform(X_test_s_text)


# %%
