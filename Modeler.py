# %%
# imports
import pandas as pd
import numpy as np

from Dataloader import data, train, val, test
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

# "Train" the model
Majority_class = Majority_model(train)


# %%
# Baseline: simple regression model