#imports
import pandas as pd
import matplotlib.pyplot as plt
import spacy
from Dataloader import data, train, val, test, corpus_0, corpus_1

# %%
#what does the data look like?
print(data.head())
print(data.shape) #(40452, 3)
print(data.isnull().sum()) #no missing data

# -----added: is this double work? 
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
#----- until here: is this double work?

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
#  Latex table for the bonus points ;)
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
plt.show()
