# %%
# imports
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import Dataloader as dl

# %%
# Baseline: Majority model

def Majority_model(data):
    """
    Input: 
    data = DataFrame with column "extrovert" for which  0 indicates 'introvert' and 1 indicates 'extrovert'.
    Output:
    0 if introvert is the majority (or tied).
    1 if extrovert is the the majority.
    """
    extrovert_count = (data.extrovert == 1).sum()
    introvert_count = (data.extrovert == 0).sum()

    if extrovert_count > introvert_count:
        return 1
    else:
        return 0

# Fitting the model
Majority_class = Majority_model(dl.train)


# %%
# Baseline: simple logistic regression model
LogisticRegression_model = LogisticRegression(max_iter=1000, class_weight="balanced")

# Fitting the model
LogisticRegression_model.fit(dl.X_train_vec, dl.y_train.values.ravel())

#%%
# --------------- Random Forest Classifier
#create classifiers
rf_10 = RandomForestClassifier(random_state=2026, 
                            criterion="entropy", 
                            n_estimators=10,
                            max_depth=20,
                            )

rf_50 = RandomForestClassifier(random_state=2026, 
                            criterion="entropy", 
                            n_estimators=50,
                            max_depth=20,
                            )

rf_100 = RandomForestClassifier(random_state=2026, 
                            criterion="entropy", 
                            n_estimators=100,
                            max_depth=20,
                            )

#fit to the training data
rf_10_random = rf_10.fit(dl.X_train_vec, dl.y_train)
rf_10_stratify = rf_10.fit(dl.X_train_s_vec, dl.y_train_s)

rf_50_random = rf_50.fit(dl.X_train_vec, dl.y_train)
rf_50_stratify = rf_50.fit(dl.X_train_s_vec, dl.y_train_s)

rf_100_random = rf_100.fit(dl.X_train_vec, dl.y_train)
rf_100_stratify = rf_100.fit(dl.X_train_s_vec, dl.y_train_s)

# %%
rf_10_random_train_score = round(rf_10_random.score(dl.X_train_vec, dl.y_train), 3)
rf_10_random_val_score = round(rf_10_random.score(dl.X_val_vec, dl.y_val), 3)

rf_50_random_train_score = round(rf_50_random.score(dl.X_train_vec, dl.y_train), 3)
rf_50_random_val_score = round(rf_50_random.score(dl.X_val_vec, dl.y_val), 3)

rf_100_random_train_score = round(rf_100_random.score(dl.X_train_vec, dl.y_train), 3)
rf_100_random_val_score = round(rf_100_random.score(dl.X_val_vec, dl.y_val), 3)

rf_10_stratify_train_score = round(rf_10_stratify.score(dl.X_train_s_vec, dl.y_train_s), 3)
rf_10_stratify_val_score = round(rf_10_stratify.score(dl.X_val_s_vec, dl.y_val_s), 3)

rf_50_stratify_train_score = round(rf_50_stratify.score(dl.X_train_s_vec, dl.y_train_s), 3)
rf_50_stratify_val_score = round(rf_50_stratify.score(dl.X_val_s_vec, dl.y_val_s), 3)

rf_100_stratify_train_score = round(rf_100_stratify.score(dl.X_train_s_vec, dl.y_train_s), 3)
rf_100_stratify_val_score = round(rf_100_stratify.score(dl.X_val_s_vec, dl.y_val_s), 3)

# %%
print(f"10 tree with random split: \ntraining accuracy = {rf_10_random_train_score} \nvalidating accuracy = {rf_10_random_val_score} \n")
print(f"50 trees with random split: \ntraining accuracy = {rf_50_random_train_score} \nvalidating accuracy = {rf_50_random_val_score} \n")
print(f"100 trees with random split: \ntraining accuracy = {rf_100_random_train_score} \nvalidating accuracy = {rf_100_random_val_score} \n")

print(f"10 tree with stratified split: \ntraining accuracy = {rf_10_stratify_train_score} \nvalidating accuracy = {rf_10_stratify_val_score} \n")
print(f"50 trees with stratified split: \ntraining accuracy = {rf_50_stratify_train_score} \nvalidating accuracy = {rf_50_stratify_val_score} \n")
print(f"100 trees with stratified split: \ntraining accuracy = {rf_100_stratify_train_score} \nvalidating accuracy = {rf_100_stratify_val_score} \n")

# %%
#random split: best = 50
#stratified split: best = 10 
from sklearn import metrics
y_pred_r = rf_50_random.predict(dl.X_test_vec)
y_pred_s = rf_10_stratify.predict(dl.X_test_s_vec)

precision_r = round(metrics.precision_score(dl.y_test, y_pred_r), 3)
recall_r = round(metrics.recall_score(dl.y_test, y_pred_r), 3)
accuracy_r = round(rf_10_random.score(dl.X_test_vec, dl.y_test), 3)
f1_r = round(metrics.f1_score(dl.y_test, y_pred_r, average="weighted"), 3)
print(f"The model with the random split has an precision of {precision_r}, a recall of {recall_r}, an accuracy of {accuracy_r}, and an f1 of {f1_r}. ")

precision_s = round(metrics.precision_score(dl.y_test_s, y_pred_s), 3)
recall_s = round(metrics.recall_score(dl.y_test_s, y_pred_s), 3)
accuracy_s = round(rf_50_stratify.score(dl.X_test_s_vec, dl.y_test_s), 3)
f1_s = round(metrics.f1_score(dl.y_test, y_pred_s, average="weighted"), 3)
print(f"The model with a stratified split has an precision of {precision_s}, a recall of {recall_s}, an accuracy of {accuracy_s}, and an f1 of {f1_s}. ")
# %%


