# Language-and-AI
Repository for the experiments described in "Decision trees for extroversion and introversion classification" for the interm assignment for JBC090 Language and AI.

## 📜 Overview

- 🔎 Paper Details
    - 🦩tl;dr
    - ♻️ Reproduction
    - 🚀 Dependencies
    - 🌱 Resources
- ⭐ Experimental manipulation


## 🔎 Paper Details


### 🦩 tl;dr

We trained Random Forest classifiers and compared their performance against baseline models, namely a majority-class model and a simple Logistic Regression model.


### ♻️ Reproduction

To reproduce the results simply run `Evaluator.py`.  For more information on how the data looks like run `Descriptives.py`. However, we cannot share the data.

> \* The code was tested with Python 3.13.3 on Windows.

### 🚀 Dependencies

The code was tested using these libraries and versions:

```
certifi==2026.1.4
charset-normalizer==3.4.4
idna==3.11
numpy==2.4.1
pandas==2.3.3
python-dateutil==2.9.0.post0
pytz==2025.2
requests==2.32.5
six==1.17.0
tzdata==2025.3
urllib3==2.6.3

```

### 🌱 Resources

Running `Evaluator.py` on the Graphic card AMD Radeon RX 7800 XT with the processor AMD Ryzen 5 7600 takes approximately five minutes.

## ⭐ Experimental manipulation

> `Dataloader.py` -- line 9
Update the file path:
```python
    data = pd.read_csv("download/extrovert_introvert.csv")
```
> `Dataloader.py` -- lines 13-14
Modify the random_state parameter to use a different seed for the data split:
```python
    train, val_test = train_test_split(data, test_size=0.2, random_state=2026)
    val, test = train_test_split(val_test, test_size=0.5, random_state=2026)
```

> `Modeler.py` -- lines 42-58
Modify the random_state parameter to use a different seed for the Random Forest models:
```python
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
```
