# %%
# Imports 
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from Dataloader import y_true_test, X_test_vec
from Modeler import Majority_class, LogisticRegression_model


# %%
# Majority model
# Predicting testing data set
y_pred_maj = np.full_like(y_true_test, Majority_class)

# Calculating Confusion matrix, accuracy, precision, recall and F1-score
cm_maj = confusion_matrix(y_true_test, y_pred_maj)
accuracy_maj = accuracy_score(y_true_test, y_pred_maj)
precision_maj = precision_score(y_true_test, y_pred_maj, zero_division=0)
recall_maj = recall_score(y_true_test, y_pred_maj, zero_division=0)
f1_maj = f1_score(y_true_test, y_pred_maj, zero_division=0)

# Printing the results
print(f"Confusion matrix of Majority model: {cm_maj}")
print(f"Accuracy of Majority model: {accuracy_maj:.3f}")
print(f"Precision of Majority model: {precision_maj:.3f}")
print(f"Recall of Majority model: {recall_maj:.3f}")
print(f"F1-score of Majority model: {f1_maj:.3f}")

# %%
# Logistic regression model
# Predicting testing data set
y_pred_log = LogisticRegression_model.predict(X_test_vec)

# Calculating Confusion matrix, accuracy, precision, recall and F1-score
cm_log = confusion_matrix(y_true_test, y_pred_log)
accuracy_log = accuracy_score(y_true_test, y_pred_log)
precision_log = precision_score(y_true_test, y_pred_log, zero_division=0)
recall_log = recall_score(y_true_test, y_pred_log, zero_division=0)
f1_log = f1_score(y_true_test, y_pred_log, zero_division=0)

# Printing the results
print(f"Confusion matrix of Logistic Regression model: {cm_log}")
print(f"Accuracy of Logistic Regression model: {accuracy_log:.3f}")
print(f"Precision of Logistic Regression model: {precision_log:.3f}")
print(f"Recall of Logistic Regression model: {recall_log:.3f}")
print(f"F1-score of Logistic Regression model: {f1_log:.3f}")