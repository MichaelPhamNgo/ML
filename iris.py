#################################################################
#
#  ML - HW 1
#
#  Martine De Cock
#
#################################################################

import pandas as pd
from sklearn import tree
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import numpy as np

# Read the dataset into a dataframe and map the labels to numbers
df = pd.read_csv('iris.csv')
map_to_int = {'setosa':0, 'versicolor':1, 'virginica':2}
df["label"] = df["species"].replace(map_to_int)
print(df)

# Separate the input features from the label
features = list(df.columns[:4])
X = df[features]
y = df["label"]

# Train a decision tree and compute its training accuracy
clf = tree.DecisionTreeClassifier(max_depth=2, criterion='entropy')
clf.fit(X, y)
print(metrics.accuracy_score(y, clf.predict(X)))

# Initialize the accuracy of the models to blank list. The accuracy of each model will be appended to this list
accuracy_model = []

kf = KFold(n_splits=10)
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model = clf.fit(X_train, y_train)
    accuracy_model.append(accuracy_score(y_test, model.predict(X_test)))
print("Accuracy: ", np.mean(accuracy_model))