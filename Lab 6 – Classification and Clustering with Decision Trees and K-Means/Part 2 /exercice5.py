import pandas as pd  
import numpy as np 
import sklearn.metrics as sm  
import matplotlib.pyplot as plt  
from sklearn.cluster import KMeans, AgglomerativeClustering 
from sklearn import datasets 

iris = datasets.load_iris()

# convert the features into a pandas dataframe
x = pd.DataFrame(iris.data)
x.columns = ['Sepal_Length', 'Sepal_width', 'Petal_Length', 'Petal_width']

y = pd.DataFrame(iris.target)
y.columns = ['Targets']

kmeans = KMeans(n_clusters=3)

kmeans.fit(x)

linkage_methods = ['single', 'average', 'complete', 'ward']
agg_clustering_results = {}

for method in linkage_methods:
    agg = AgglomerativeClustering(n_clusters=3, linkage=method)
    agg_labels = agg.fit_predict(x)
    score = sm.adjusted_rand_score(y['Targets'], agg_labels)
    agg_clustering_results[method] = score
    print(f"Adjusted Rand Index for Agglomerative Clustering with {method} linkage: {score:.4f}")

# K-means adjusted rand score
kmeans_score = sm.adjusted_rand_score(y['Targets'], kmeans.labels_)
print(f"Adjusted Rand Index for K-means: {kmeans_score:.4f}")