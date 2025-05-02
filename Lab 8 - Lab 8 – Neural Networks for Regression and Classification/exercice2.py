import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelBinarizer, scale
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Chargement des données IRIS et transformation
iris = load_iris()
X = iris.data  # 4 colonnes de caractéristiques
y = iris.target  # 5ème colonne : classes (0, 1, 2)

# 2. Normalisation des données (centrées – réduites)
X_scaled = scale(X)

# 3. Binarisation des labels (cible)
lb = LabelBinarizer()
y_binarized = lb.fit_transform(y)

# 4. Initialisation du MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(2,), activation='logistic', 
                    learning_rate_init=0.2, solver='lbfgs', max_iter=200)

# 5. Apprentissage avec le MLP
mlp.fit(X_scaled, y)

# 6. Prédiction et évaluation
y_pred = mlp.predict(X_scaled)
accuracy = accuracy_score(y, y_pred)

# 7. Affichage des résultats
print("\nClassification Report :\n", classification_report(y, y_pred))
print("Précision du modèle :", accuracy)