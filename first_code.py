#%% run only once, at the start
#imports
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#load data
data = pd.read_csv("download/extrovert_introvert.csv")

# %%
#what does the data look like?
print(data.head())
print(data.shape) #(40452, 3)
print(data.isnull().sum()) #no missing data
# %% 
# split dataset 80/10/10 with all sets having equal density int/ex (stratified split)
# use random state for reproducibility
train, val_test = train_test_split(data, test_size=0.2, stratify=data.extrovert, random_state=2026)
val, test = train_test_split(val_test, test_size=0.5, stratify=val_test.extrovert, random_state=2026)

#plot to show equal splits
plt.hist([train.extrovert, val.extrovert, test.extrovert], density=True)
plt.legend(['train', 'validation', 'test'])
plt.title("Stratified split", fontsize=15)
plt.xlabel('Extrovert', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.xticks([0, 1]);

#print size of each set
print("Training size: ", train.shape[0])
print("Validating size: ", val.shape[0])
print("Testing size: ", test.shape[0])
# %%
#next, split randomly 80/10/10
train, val_test = train_test_split(data, test_size=0.2, random_state=2026)
val, test = train_test_split(val_test, test_size=0.5, random_state=2026)

plt.hist([train.extrovert, val.extrovert, test.extrovert], density=True)
plt.legend(['train', 'validation', 'test'])
plt.title("Stratified split", fontsize=15)
plt.ylabel('Density', fontsize=12)
plt.xticks([0, 1], ["Introvert", "Extrovert"], fontsize=12);


#print size of each set
print("Training size: ", train.shape[0])
print("Validating size: ", val.shape[0])
print("Testing size: ", test.shape[0])
print('Extrovert train count: %d'  %(train.extrovert == 1).sum()) 
print('Extrovert val count: %d'  %(val.extrovert == 1).sum()) 
print('Extrovert test count: %d'  %(test.extrovert == 1).sum()) 
print(round(test.extrovert == 1).sum() / len(data) * 100)


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
#this makes two separate lists for posts from extroverts and introverts
corpus_0 = []
corpus_1 = []
for index, row in data.iterrows():
    if row.extrovert == 0:
        corpus_0.append(row.post)
    else:
        corpus_1.append(row.post)