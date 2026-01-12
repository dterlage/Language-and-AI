# %%
# Imports 
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from Dataloader import y_true_test
from Modeler import Majority_class


# %%
# Majority model
y_pred_maj = np.full_like(y_true_test, Majority_class)

# Calculating Confusion matrix, accuracy, precision, recall and F1-score
cm_maj = confusion_matrix(y_true_test, y_pred_maj)
accuracy_maj = accuracy_score(y_true_test, y_pred_maj)
precision_maj = precision_score(y_true_test, y_pred_maj)
recall_maj = recall_score(y_true_test, y_pred_maj)
f1_maj = f1_score(y_true_test, y_pred_maj)

# Printing the results
print(f"Confusion matrix of Majority model: {cm_maj}")
print(f"Accuracy of Majority model: {accuracy_maj:.3f}")
print(f"Precision of Majority model: {precision_maj:.3f}")
print(f"Recall of Majority model: {recall_maj:.3f}")
print(f"F1-score of Majority model: {f1_maj:.3f}")

# %%
# Accuracy, recall, precision and F1-score