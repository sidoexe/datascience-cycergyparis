import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

# 1. Génération des données
x = np.linspace(0, 8, 20).reshape(-1, 1)  # 20 points entre 0 et 8
y = np.sin(x).reshape(-1) + np.random.randn(20)*0.2  # Ajouter du bruit gaussien

# Vérification des tailles
print("Taille de x:", x.shape)
print("Taille de y:", y.shape)

# 2. Visualisation des points d'apprentissage
plt.scatter(x, y, color='red', label='Points d\'apprentissage')
plt.title('Données avec bruit')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

# 3. Initialisation du MLP
mlp = MLPRegressor(hidden_layer_sizes=(3,), activation='tanh', 
                   learning_rate_init=0.1, max_iter=100)

# 4. Apprentissage avec le MLP
mlp.fit(x, y)

# 5. Prédictions avec le MLP appris
y_pred = mlp.predict(x)

# 6. Tracer la fonction sinusoïdale réelle et la fonction estimée par le MLP
plt.scatter(x, y, color='red', label='Points d\'apprentissage')
plt.plot(x, np.sin(x), label='Fonction sinusoïdale réelle', color='blue')
plt.plot(x, y_pred, label='Fonction estimée par le MLP', color='green')
plt.title('Régression avec un Perceptron Multi-Couches')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()