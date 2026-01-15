# %%
# Imports 
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

import Dataloader as dl
import Modeler as m

# %%
# Majority model
# Predicting testing data set
y_pred_maj = np.full_like(dl.y_true_test, m.Majority_class)

# Calculating Confusion matrix, accuracy, precision, recall and F1-score
cm_maj = confusion_matrix(dl.y_true_test, y_pred_maj)
accuracy_maj = accuracy_score(dl.y_true_test, y_pred_maj)
precision_maj = precision_score(dl.y_true_test, y_pred_maj, zero_division=0)
recall_maj = recall_score(dl.y_true_test, y_pred_maj, zero_division=0)
f1_maj = f1_score(dl.y_true_test, y_pred_maj, zero_division=0)

# Printing the results
print(f"Confusion matrix of Majority model: {cm_maj}")
print(f"Accuracy of Majority model: {accuracy_maj:.3f}")
print(f"Precision of Majority model: {precision_maj:.3f}")
print(f"Recall of Majority model: {recall_maj:.3f}")
print(f"F1-score of Majority model: {f1_maj:.3f}")

# %%
# Logistic regression model
# Predicting testing data set
y_pred_log = m.LogisticRegression_model.predict(dl.X_test_vec)

# Calculating Confusion matrix, accuracy, precision, recall and F1-score
cm_log = confusion_matrix(dl.y_true_test, y_pred_log)
accuracy_log = accuracy_score(dl.y_true_test, y_pred_log)
precision_log = precision_score(dl.y_true_test, y_pred_log, zero_division=0)
recall_log = recall_score(dl.y_true_test, y_pred_log, zero_division=0)
f1_log = f1_score(dl.y_true_test, y_pred_log, zero_division=0)

# Printing the results
print(f"Confusion matrix of Logistic Regression model: {cm_log}")
print(f"Accuracy of Logistic Regression model: {accuracy_log:.3f}")
print(f"Precision of Logistic Regression model: {precision_log:.3f}")
print(f"Recall of Logistic Regression model: {recall_log:.3f}")
print(f"F1-score of Logistic Regression model: {f1_log:.3f}")


#--------Random forest
rf_1_random_train_score = round(m.rf_1_random.score(dl.X_train_vec, dl.y_train), 3)
rf_1_random_val_score = round(m.rf_1_random.score(dl.X_val_vec, dl.y_val), 3)

rf_5_random_train_score = round(m.rf_5_random.score(dl.X_train_vec, dl.y_train), 3)
rf_5_random_val_score = round(m.rf_5_random.score(dl.X_val_vec, dl.y_val), 3)

rf_10_random_train_score = round(m.rf_10_random.score(dl.X_train_vec, dl.y_train), 3)
rf_10_random_val_score = round(m.rf_10_random.score(dl.X_val_vec, dl.y_val), 3)

rf_1_stratify_train_score = round(m.rf_1_stratify.score(dl.X_train_s_vec, dl.y_train_s), 3)
rf_1_stratify_val_score = round(m.rf_1_stratify.score(dl.X_val_s_vec, dl.y_val_s), 3)

rf_5_stratify_train_score = round(m.rf_5_stratify.score(dl.X_train_s_vec, dl.y_train_s), 3)
rf_5_stratify_val_score = round(m.rf_5_stratify.score(dl.X_val_s_vec, dl.y_val_s), 3)

rf_10_stratify_train_score = round(m.rf_10_stratify.score(dl.X_train_s_vec, dl.y_train_s), 3)
rf_10_stratify_val_score = round(m.rf_10_stratify.score(dl.X_val_s_vec, dl.y_val_s), 3)

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
y_pred_r = m.rf_5_random.predict(dl.X_test_vec)
y_pred_s = m.rf_1_stratify.predict(dl.X_test_s_vec)

precision_r = round(metrics.precision_score(dl.y_test, y_pred_r), 3)
recall_r = round(metrics.recall_score(dl.y_test, y_pred_r), 3)
accuracy_r = round(m.rf_1_random.score(dl.X_test_vec, dl.y_test), 3)
f1_r = round(metrics.f1_score(dl.y_test, y_pred_r, average="weighted"), 3)
print(f"The model with the random split has an precision of {precision_r}, a recall of {recall_r}, an accuracy of {accuracy_r}, and an f1 of {f1_r}. ")

precision_s = round(metrics.precision_score(dl.y_test_s, y_pred_s), 3)
recall_s = round(metrics.recall_score(dl.y_test_s, y_pred_s), 3)
accuracy_s = round(m.rf_5_stratify.score(dl.X_test_s_vec, dl.y_test_s), 3)
f1_s = round(metrics.f1_score(dl.y_test, y_pred_s, average="weighted"), 3)
print(f"The model with a stratified split has an precision of {precision_s}, a recall of {recall_s}, an accuracy of {accuracy_s}, and an f1 of {f1_s}. ")
# %%