# DataScience-CyCergyParis

This repository contains a collection of data science labs completed during my studies at **CY Cergy Paris**. Each lab explores key concepts in data analysis, machine learning, and statistical modeling. The work demonstrates both theoretical understanding and practical applications using Python, NumPy, pandas, scikit-learn, and matplotlib.

## Labs Overview

### Lab 1 – Data Quality Analysis with Pandas
This lab focuses on exploring core functionalities of the Pandas library and applying data cleaning techniques to a real-world dataset (jobs.csv). It includes detection and correction of missing values, format inconsistencies, and generation of a cleaned DataFrame for analysis.

**Tools:** `pandas`

---

### Lab 2 – Data Visualization
This lab introduces the fundamentals of data visualization using Python. 

In Part 1, we analyze and comment on a provided visualization script (vizDFVerENT.py) to understand how it processes and displays data. 

In Part 2, we create various plots from the company_sales_data.csv dataset, including line charts, bar plots, and distribution graphs, to explore sales trends, product comparisons, and cumulative figures. The lab emphasizes the importance of choosing appropriate visualization techniques to effectively communicate insights.

**Tools:** `pandas`, `matplotlib`, `numpy`

---

### Lab 3 – Clustering Techniques and Distance Metrics
This lab focuses on clustering methods and similarity measures used in data analysis. 

In the first part, we compute and compare the Jaccard similarity coefficient, Jaccard weighted coefficient, and angular distance for binary data representing patients’ sugar and salt levels. 

In the second part, we apply clustering algorithms on a 2D dataset using Manhattan distance. The methods explored include K-means (with specified initial centroids) and Agglomerative Clustering with single, complete, and average linkage strategies.

 The goal is to understand how distance metrics and linkage choices impact clustering outcomes. 

---

### Lab 4 – Decision Tree Classification and Rule Extraction
This lab introduces decision tree classification using a small dataset related to weather conditions and the decision to play tennis. 

The task involves building a decision tree based on features such as outlook, temperature, humidity, and wind, and deriving a set of if-then classification rules from the resulting tree. 

The goal is to understand how decision trees model decisions and how rules can be extracted to interpret model behavior.

---

### Lab 5 - Association Mining
This lab explores the fundamentals of association rule mining, focusing on identifying relationships between items in transactional data. 

Using a minimum support of 40% and a minimum confidence of 75%, we extract frequent itemsets and generate strong association rules. 

The exercise emphasizes understanding the concepts of support, confidence, and the relevance of discovered patterns in data-driven decision making.

---

### Lab 6 – Classification and Clustering with Decision Trees and K-Means
This lab explores supervised and unsupervised learning through a series of classification and clustering tasks. 

Using Python libraries such as scikit-learn, i implemented Decision Tree classifiers on datasets like balance-scale.data and JoggingTitre.csv, observing the effects of model parameters like max_depth and applying K-Fold cross-validation. 

For clustering, the lab applied K-Means and Agglomerative Clustering (with various linkage strategies) to structured 1D datasets and the Iris dataset, including performance evaluations using metrics like adjusted_rand_score. 

The second part consisted of in-depth experiments (e.g., training size vs. accuracy, linkage method comparisons) on datasets like car.data and tic-tac-toe.data, supported by plots and analytical observations.

**Tools:** `scikit-learn`, `pandas`, `numpy`, `graphviz`

---

### Lab 7 – Introduction to Machine Learning with KNN and Model Comparison
This lab introduced the principles of Machine Learning, focusing on supervised classification methods. 

It began with a hands-on implementation of the K-Nearest Neighbor (KNN) algorithm to highlight how distance-based learning works. i explored the effect of different values of K, using leave-one-out cross-validation to measure prediction error and compare results with those from scikit-learn’s built-in KNN implementation.

In the second part, i applied and compared three core machine learning algorithms—Support Vector Machines (SVMs), Naive Bayes, and Decision Trees—on the well-known Iris dataset. 

I evaluated performance using confusion matrices and error rates, gaining insights into the strengths and weaknesses of each algorithm. 

The lab also introduced basic concepts of model evaluation and tuning, reinforcing key foundations of Machine Learning.

**Tools:** `scikit-learn`, `numpy`

---

### Lab 8 – Neural Networks for Regression and Classification
This lab marked a deeper dive into Machine Learning, focusing on the fundamentals of Neural Networks using the Multi-Layer Perceptron (MLP) architecture from scikit-learn. 

The first part involved applying an MLP to perform regression on a noisy sinusoidal dataset, allowing me to understand how neural networks approximate continuous functions. By experimenting with key hyperparameters like the number of hidden neurons, activation functions, learning rate, and number of iterations, students explored how these settings impact model performance.

In the second part, i used an MLP for classification on the Iris dataset, including preprocessing steps like label binarization and feature scaling.

Visualization techniques such as PCA (Principal Component Analysis) helped analyze classification results. By testing different optimization algorithms and hyperparameter settings, i observed the sensitivity of neural networks to data normalization and architectural choices, reinforcing important concepts in neural network training and evaluation.

**Tools:** `scikit-learn`, `numpy`, `matplotlib`

---

### Acknowledgments

I would like to express my sincere gratitude to my professors, **Grozavu**, **Nistor**, **Jen**, **Tao Yuan**, and **Dimitris Kotzinos**, for their invaluable guidance and support throughout my data science studies at CY Cergy Paris. Their expertise and encouragement have been instrumental in my academic development and in the successful completion of these labs. 

Thank you for your dedication and for making the learning experience so enriching!
