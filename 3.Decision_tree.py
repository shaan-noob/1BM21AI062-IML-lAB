import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
-------------------------------
from sklearn.datasets import load_breast_cancer

# Load the breast cancer dataset
dataset = load_breast_cancer()

# Access features (X) and target variable (y)
X = dataset.data
y = dataset.target

#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
-----------------------------
#Training the Decision Tree Classification model on the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(
    class_weight=None,
    criterion='entropy',
    max_depth=None,
    max_features=None,
    max_leaf_nodes=None,
    min_samples_leaf=1,
    min_samples_split=2,
    random_state=0,
    splitter='best'
)
--------------------------
classifier.fit(X_train, y_train)

#Display the results (confusion matrix and accuracy)
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
