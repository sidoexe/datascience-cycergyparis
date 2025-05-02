import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.model_selection import KFold

# this line reads the data from the file
balance_data = pd.read_csv('balance-scale.data', sep= ',', header= None)

# this line selects all rows and columns 1 through 4 and stores it in X
X = balance_data.values[:, 1:5]

# this line selects all rows and column 0 and stores it in Y
Y = balance_data.values[:,0]

# set up 10-fold cross-validation, shuffling data with a random seed of 10
kfold = KFold(n_splits=10, shuffle=True, random_state=10)

# initialize accuracy tracking variables
ac=0.0
ac_score=0.0

# initialize the DecisionTreeClassifier with entropy criterion
clf_entropy = DecisionTreeClassifier(criterion='entropy')

# loop through each train-test split created by kfold
for train, test in kfold.split(X):
        
    # split data into training and test sets
    X_train, X_test, y_train, y_test = X[train], X[test], Y[train], Y[test]
    
    # train the classifier using the entropy criterion
    clf_entropy.fit(X_train, y_train)
    
    # make predictions on the test set
    y_pred_en = clf_entropy.predict(X_test)
    
    # calculate accuracy for this split and add it to the total
    ac_score=accuracy_score(y_test,y_pred_en)*100
    ac=ac+ac_score
    
    # print the accuracy of this split
    print ("Accuracy is ", ac_score)

# calculate and print the average accuracy across all 10 splits    
ac_avg=ac/10
print ("Average Accuracy is ", ac_avg)