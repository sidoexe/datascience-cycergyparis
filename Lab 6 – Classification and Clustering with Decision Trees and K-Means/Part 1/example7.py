import pandas as pd  
import numpy as np 
import sklearn.metrics as sm  
import matplotlib.pyplot as plt  
from sklearn.cluster import KMeans 
from sklearn import datasets 
from sklearn import metrics

# load the iris dataset
iris = datasets.load_iris()

# print the dataset details
print(iris)

# print the data of the iris dataset
print(iris.data)

# print the feature names
print(iris.feature_names)

# print the target values
print(iris.target)

# print the target names
print(iris.target_names)

# convert the features into a pandas dataframe
x = pd.DataFrame(iris.data)

# rename the columns for better readability
x.columns = ['Sepal_Length', 'Sepal_width', 'Petal_Length', 'Petal_width']

# convert the target labels into a pandas dataframe
y = pd.DataFrame(iris.target)
y.columns = ['Targets']

# initialize the k-means model with 3 clusters
model = KMeans(n_clusters=3)

# fit the k-means model to the data
model.fit(x)

# define a colormap for visualization, i went for Red, Green, Blue
colormap = np.array(['r', 'g', 'b'])

# create a scatter plot of Petal Length vs Petal Width, color-coded by the true target values
plt.scatter(x.Petal_Length, x.Petal_width, c=colormap[y.Targets], s=40)

# display the plot
plt.show()

# create another scatter plot of Petal Length vs Petal Width, color-coded by the predicted labels
plt.scatter(x.Petal_Length, x.Petal_width, c=colormap[model.labels_], s=40)

# display the second plot
plt.show()