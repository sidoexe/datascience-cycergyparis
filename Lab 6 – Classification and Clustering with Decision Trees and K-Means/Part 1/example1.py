import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

# this line reads the data from the file
balance_data = pd.read_csv('balance-scale.data', sep= ',', header= None)

# this line retrieves the shape of the data (rows, columns)
balance_data.shape

# this line selects all rows and columns 1 through 4 and stores it in X
X = balance_data.values[:, 1:5]

# this line selects all rows and column 0 and stores it in Y
Y = balance_data.values[:,0]

# this line splits the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3)

# this line creates a decision tree classifier
clf_entropy = DecisionTreeClassifier(criterion = "entropy", max_depth=3,min_samples_leaf=5)

# this line fits the classifier to the training data
clf_entropy.fit(X_train, y_train)

# this line predicts the values of the testing data
y_pred_en = clf_entropy.predict(X_test)

# this line prints the accuracy of the model
print ("Accuracy is ", accuracy_score(y_test,y_pred_en)*100)