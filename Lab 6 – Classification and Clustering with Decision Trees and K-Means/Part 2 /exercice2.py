import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


car_data = pd.read_csv('car.data', header=None)
ttt_data = pd.read_csv('tic-tac-toe.data', header=None)

# encode categorical features to numerical values
encoder = LabelEncoder()
for col in car_data.columns:
    car_data[col] = encoder.fit_transform(car_data[col])

for col in ttt_data.columns:
    ttt_data[col] = encoder.fit_transform(ttt_data[col])

# split data into features and labels
X_car = car_data.iloc[:, :-1]
y_car = car_data.iloc[:, -1]
X_ttt = ttt_data.iloc[:, :-1]
y_ttt = ttt_data.iloc[:, -1]

# list of training data sizes
sizes = [0.1, 0.25, 0.33, 0.5, 0.66, 0.75]

def random_subsampling_decision_tree(X, y, dataset_name):
    """Function to perform random subsampling and calculate average accuracy"""
    print(f"Evaluating {dataset_name} dataset")
    avg_accuracies = []
    
    for train_size in sizes:
        accuracies = []
        for _ in range(10):
            # split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=None)
            
            # train Decision Tree
            clf = DecisionTreeClassifier()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            accuracies.append(accuracy_score(y_test, y_pred))
        
        avg_accuracy = np.mean(accuracies)
        avg_accuracies.append(avg_accuracy)
        print(f"Training Size: {train_size * 100}%, Average Accuracy: {avg_accuracy:.2f}")
    
    # plot accuracy vs training size
    plt.plot([size * 100 for size in sizes], avg_accuracies, marker='o', label=dataset_name)
    plt.xlabel('Training Data Size (%)')
    plt.ylabel('Average Accuracy')
    plt.title(f'Decision Tree Performance for {dataset_name} Dataset')
    plt.legend()
    plt.grid(True)

random_subsampling_decision_tree(X_car, y_car, "Car")
random_subsampling_decision_tree(X_ttt, y_ttt, "Tic-Tac-Toe")

plt.show()