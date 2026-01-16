#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import spacy
from Dataloader import data, train, val, test, corpus_0, corpus_1, train_s, val_s, test_s

# %%
#what does the data look like?
print(data.head())
print(data.shape) #(40452, 3)
print(data.isnull().sum()) #no missing data

tokens_0 = []
tokens_1 = []
nlp = spacy.blank("en")

#tokenizing per class
for post in nlp.pipe(corpus_0, batch_size=1000):
    tokens_0.extend(token.lower_ for token in post)
print("d")
for post in nlp.pipe(corpus_1, batch_size=1000):
    tokens_1.extend(token.lower_ for token in post)

print("Total tokens in class 0: ", len(tokens_0))
print("Total tokens in class 1: ", len(tokens_1))

tokens_both = set(tokens_0) & set(tokens_1)
only_0 = set(tokens_0) - set(tokens_1)
only_1 = set(tokens_1) - set(tokens_0)

print("Shared tokens:", len(tokens_both))
print("Only in class 0:", len(only_0))
print("Only in class 1:", len(only_1))

print("Unique tokens in class 0:", len(tokens_both)+ len(only_0))
print("Unique tokens in class 1:", len(tokens_both)+ len(only_1))

# %%
split_info = {
    "Data sets" : ["Train", "Validation", "Test"],
    "Size" : [train.shape[0], val.shape[0], test.shape[0]],
    "Extrovert count" : [(train.extrovert == 1).sum(), (val.extrovert == 1).sum(), (test.extrovert == 1).sum()],
    "Introvert count" : [(train.extrovert == 0).sum(), (val.extrovert == 0).sum(), (test.extrovert == 0).sum()]
    }
df_split = pd.DataFrame(split_info)
print(df_split.head())

# %%
# Latex table: https://www.tilburgsciencehub.com/topics/visualization/reporting-tables/reportingtables/pandas-latex-tables/
# \usepackage{booktabs}
latex_table_split = df_split.to_latex(
    index=False,  # To not include the DataFrame index as a column in the table
    caption="Comparison of the train, validation and test data sets",  # The caption to appear above the table in the LaTeX document
    label="tab:split_comparison",  # A label used for referencing the table within the LaTeX document
    position="htbp",  # The preferred positions where the table should be placed in the document ('here', 'top', 'bottom', 'page')
    column_format="lccc",  # The format of the columns: left-aligend first column and center-aligned remaining columns as per APA guidelines
    escape=False,  # Disable escaping LaTeX special characters in the DataFrame
    float_format="{:0.2f}".format  # Formats floats to two decimal places
)

print(latex_table_split)

# %%
# Visualize the splits
plt.hist([train.extrovert, val.extrovert, test.extrovert], density=True)
plt.legend(['train', 'validation', 'test'])
plt.title("Stratified split", fontsize=15)
plt.ylabel('Density', fontsize=12)
plt.xticks([0, 1], ["Introvert", "Extrovert"], fontsize=12)
#plt.show()

# %%
# Visualize the data sets with a bar chart
sets = ["Train", "Train Stratified", "Validation", "Validation Stratified", "Test", "Test Stratified"]
sets_info = {
    "Size" : [train.shape[0], train_s.shape[0], val.shape[0], val_s.shape[0], test.shape[0],  test_s.shape[0]],
    "Extrovert count" : [(train.extrovert == 1).sum(),(train_s.extrovert == 1).sum(), (val.extrovert == 1).sum(),
                         (val_s.extrovert == 1).sum(), (test.extrovert == 1).sum(), (test_s.extrovert == 1).sum()],
    "Introvert count" : [(train.extrovert == 0).sum(), (train_s.extrovert == 0).sum(), (val.extrovert == 0).sum(),
                         (val_s.extrovert == 0).sum(), (test.extrovert == 0).sum(), (test_s.extrovert == 0).sum()]

}

x = np.arange(len(sets))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for attribute, measurement in sets_info.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Count')
ax.set_title('Data set splits')
ax.set_xticks(x + width, sets)
ax.legend(loc='upper left', ncols=3)
ax.set_ylim(0, 40000)
plt.xticks(rotation=45)

plt.show()

