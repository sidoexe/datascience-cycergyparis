import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import graphviz

# this line reads the data from the file
Jogging_data=pd.read_csv('JoggingTitre.csv', sep=',')

# this line extracts the column 'Jogging' and stores it in y
y=Jogging_data['Jogging']

# extract all columns except 'jogging' and store them in x
x=Jogging_data.drop(['Jogging'], axis=1)

# convert categorical columns into binary columns
x_dum=pd.get_dummies(x)

# create the decision tree classifier using entropy as the splitting criterion
clf_entropy = DecisionTreeClassifier(criterion = "entropy")

# train the model using the transformed features (x_dum) and target (y)
outputTree=clf_entropy.fit(x_dum, y)

# export the decision tree in dot format (text format for graphviz)
dot_data = tree.export_graphviz(outputTree, out_file=None)

# create a graphviz object and generate a pdf file named "Td2_dum01"
graph = graphviz.Source(dot_data)
graph.render("Td2_dum01")

# export the decision tree again, this time with feature names for better readability
dot_data = tree.export_graphviz(outputTree, out_file=None, feature_names =
x_dum.columns)

# create a new graphviz object and generate a pdf file named "Td2_dum01Name" with feature names
graph = graphviz.Source(dot_data)
graph.render("Td2_dum01Name")

#Check and compare the files Td2_dum01.pdf and Td2_dum01Name.pdf