import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Charger les données
data_path = './data/Student_Performance.csv'
data = pd.read_csv(data_path)

# Prétraitement des données
data['Extracurricular Activities'] = data['Extracurricular Activities'].map({'Yes': 1, 'No': 0})  # Encoder les activités extrascolaires

# Séparer les caractéristiques (X) et la cible (y)
X = data[['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced']]
y = data['Performance Index']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialiser le modèle de régression linéaire
model = LinearRegression()

# Validation croisée (5 plis) pour évaluer le modèle
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=kf)

# Calcul du RMSE global
rmse_global = np.sqrt(-cv_scores.mean())
print(f"RMSE global (validation croisée) : {rmse_global}")

# Entraîner le modèle sur l'ensemble d'entraînement complet
model.fit(X_train, y_train)

# Afficher les coefficients de régression
coefficients = pd.DataFrame({
    'Variable': X.columns,
    'Coefficient': model.coef_
}).sort_values(by='Coefficient', ascending=False)
print("\nCoefficients de régression :")
print(coefficients)

# Identifier les variables les plus influentes
print("\nVariable la plus influente :")
print(coefficients.iloc[0])

# Prédictions sur l'ensemble de test
y_pred = model.predict(X_test)

# Calcul du RMSE sur l'ensemble de test
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"\nRMSE sur l'ensemble de test : {rmse_test}")

# Graphique des valeurs réelles vs valeurs prédites
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred, label='Valeurs prédites', alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', label='Droite identité')
plt.xlabel("Valeurs réelles")
plt.ylabel("Valeurs prédites")
plt.title("Valeurs réelles vs Valeurs prédites")
plt.legend()
plt.grid()
plt.show()

# Rôle de la droite identité
print("\nRôle de la droite identité :")
print("La droite identité représente les cas où les valeurs prédites sont exactement égales aux valeurs réelles.")
print("Dans ce graphique, plus les points sont proches de cette droite, meilleur est le modèle.")
