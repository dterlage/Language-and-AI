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
rf_1 = RandomForestClassifier(random_state=2026, 
                            criterion="entropy", 
                            n_estimators=10,
                            max_depth=20,
                            )

rf_5 = RandomForestClassifier(random_state=2026, 
                            criterion="entropy", 
                            n_estimators=50,
                            max_depth=20,
                            )

rf_10 = RandomForestClassifier(random_state=2026, 
                            criterion="entropy", 
                            n_estimators=100,
                            max_depth=20,
                            )

#fit to the training data
rf_1_random = rf_1.fit(dl.X_train_vec, dl.y_train)
rf_1_stratify = rf_1.fit(dl.X_train_s_vec, dl.y_train_s)

rf_5_random = rf_5.fit(dl.X_train_vec, dl.y_train)
rf_5_stratify = rf_5.fit(dl.X_train_s_vec, dl.y_train_s)

rf_10_random = rf_10.fit(dl.X_train_vec, dl.y_train)
rf_10_stratify = rf_10.fit(dl.X_train_s_vec, dl.y_train_s)

# %%
rf_1_random_train_score = round(rf_1_random.score(dl.X_train_vec, dl.y_train), 3)
rf_1_random_val_score = round(rf_1_random.score(dl.X_val_vec, dl.y_val), 3)

rf_5_random_train_score = round(rf_5_random.score(dl.X_train_vec, dl.y_train), 3)
rf_5_random_val_score = round(rf_5_random.score(dl.X_val_vec, dl.y_val), 3)

rf_10_random_train_score = round(rf_10_random.score(dl.X_train_vec, dl.y_train), 3)
rf_10_random_val_score = round(rf_10_random.score(dl.X_val_vec, dl.y_val), 3)

rf_1_stratify_train_score = round(rf_1_stratify.score(dl.X_train_s_vec, dl.y_train_s), 3)
rf_1_stratify_val_score = round(rf_1_stratify.score(dl.X_val_s_vec, dl.y_val_s), 3)

rf_5_stratify_train_score = round(rf_5_stratify.score(dl.X_train_s_vec, dl.y_train_s), 3)
rf_5_stratify_val_score = round(rf_5_stratify.score(dl.X_val_s_vec, dl.y_val_s), 3)

rf_10_stratify_train_score = round(rf_10_stratify.score(dl.X_train_s_vec, dl.y_train_s), 3)
rf_10_stratify_val_score = round(rf_10_stratify.score(dl.X_val_s_vec, dl.y_val_s), 3)

# %%
print(f"1 tree with random split: \ntraining accuracy = {rf_1_random_train_score} \nvalidating accuracy = {rf_1_random_val_score} \n")
print(f"5 trees with random split: \ntraining accuracy = {rf_5_random_train_score} \nvalidating accuracy = {rf_5_random_val_score} \n")
print(f"10 trees with random split: \ntraining accuracy = {rf_10_random_train_score} \nvalidating accuracy = {rf_10_random_val_score} \n")

print(f"1 tree with stratified split: \ntraining accuracy = {rf_1_stratify_train_score} \nvalidating accuracy = {rf_1_stratify_val_score} \n")
print(f"5 trees with stratified split: \ntraining accuracy = {rf_5_stratify_train_score} \nvalidating accuracy = {rf_5_stratify_val_score} \n")
print(f"10 trees with stratified split: \ntraining accuracy = {rf_10_stratify_train_score} \nvalidating accuracy = {rf_10_stratify_val_score} \n")

# %%
#random split: best = 5 
#stratified split: best = 1 
from sklearn import metrics
y_pred_logreg = rf_5_random.predict(dl.X_test_vec)
y_pred_tree = rf_1_stratify.predict(dl.X_test_s_vec)

precision_logreg = round(metrics.precision_score(dl.y_test, y_pred_logreg), 3)
recall_logreg = round(metrics.recall_score(dl.y_test, y_pred_logreg), 3)
accuracy_logreg = round(rf_1_random.score(dl.X_test_vec, dl.y_test), 3)
f1_logreg = round(metrics.f1_score(dl.y_test, y_pred_logreg, average="weighted"), 3)
print(f"The model with the random split has an precision of {precision_logreg}, a recall of {recall_logreg}, an accuracy of {accuracy_logreg}, and an f1 of {f1_logreg}. ")

precision_tree = round(metrics.precision_score(dl.y_test_s, y_pred_tree), 3)
recall_tree = round(metrics.recall_score(dl.y_test_s, y_pred_tree), 3)
accuracy_tree = round(rf_5_stratify.score(dl.X_test_s_vec, dl.y_test_s), 3)
f1_tree = round(metrics.f1_score(dl.y_test, y_pred_tree, average="weighted"), 3)
print(f"The model with a stratified split has an precision of {precision_tree}, a recall of {recall_tree}, an accuracy of {accuracy_tree}, and an f1 of {f1_tree}. ")
# %%

