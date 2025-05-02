import numpy as np 
from sklearn.cluster import AgglomerativeClustering 
from sklearn.cluster import KMeans 

# create a dataset of 1-dimensional points
X = np.array([[1], [2], [3], [6], [7], [8], [13], [15], [17]])

# initialize the k-means model with 3 clusters, custom centers, and n_init explicitly set to 1
kmeans = KMeans(n_clusters=3, init=np.array([[1], [2], [3]]), n_init=1)

# fit the k-means model to the data
kmeans.fit(X)

# get the coordinates of the cluster centers
print("kmeans cluster centers:", kmeans.cluster_centers_)

# get the cluster labels assigned to each data point
print("kmeans labels:", kmeans.labels_)

# initialize an agglomerative clustering model with 3 clusters using single linkage
cluster = AgglomerativeClustering(n_clusters=3, linkage='single')

# fit the agglomerative clustering model to the data
cluster.fit(X)

# get the cluster labels assigned to each data point
print("agglomerative labels:", cluster.labels_)