#%% run only once, at the start
#imports
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#load data
data = pd.read_csv("download/extrovert_introvert.csv")

# %% 
# split dataset 80/10/10 with all sets having equal density int/ex (stratified split)
# use random state for reproducibility
train, val_test = train_test_split(data, test_size=0.2, stratify=data.extrovert, random_state=2026)
val, test = train_test_split(val_test, test_size=0.5, stratify=val_test.extrovert, random_state=2026)

# %%
#next, split randomly 80/10/10
train, val_test = train_test_split(data, test_size=0.2, random_state=2026)
val, test = train_test_split(val_test, test_size=0.5, random_state=2026)

 # %%
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

