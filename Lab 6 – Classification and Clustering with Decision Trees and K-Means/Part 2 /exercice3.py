import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import make_blobs

# Generate sample data
X, _ = make_blobs(n_samples=10, centers=3, cluster_std=1.0, random_state=42)

# Compute the linkage matrix using different linkage methods
linkage_methods = ['single', 'average', 'complete']

plt.figure(figsize=(12, 8))

for i, method in enumerate(linkage_methods):
    plt.subplot(1, 3, i + 1)
    Z = linkage(X, method=method)
    dendrogram(Z)
    plt.title(f'({method.capitalize()} Linkage)')

plt.tight_layout()
plt.show()