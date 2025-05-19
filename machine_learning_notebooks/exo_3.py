import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


data_path = './data/Student_Performance.csv'
data = pd.read_csv(data_path)

data['Extracurricular Activities'] = data['Extracurricular Activities'].map({'Yes': 1, 'No': 0})

X = data[['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced']]
y = data['Performance Index']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
k_values = range(1, 21)
train_errors = []
val_errors = []

for k in k_values:
    knn = KNeighborsRegressor(n_neighbors=k)
    train_scores = []
    val_scores = []
    for train_idx, val_idx in kf.split(X_train):
        X_t, X_v = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_t, y_v = y_train.iloc[train_idx], y_train.iloc[val_idx]
        knn.fit(X_t, y_t)
        train_scores.append(mean_squared_error(y_t, knn.predict(X_t)))
        val_scores.append(mean_squared_error(y_v, knn.predict(X_v)))
    train_errors.append(np.mean(train_scores))
    val_errors.append(np.mean(val_scores))


plt.figure()
plt.plot(k_values, train_errors, label='Erreur d\'entraînement')
plt.plot(k_values, val_errors, label='Erreur de validation')
plt.xlabel('Nombre de voisins (k)')
plt.ylabel('Erreur quadratique moyenne')
plt.legend()
plt.title('Erreur d\'entraînement et de validation en fonction de k')
plt.show()

optimal_k = k_values[np.argmin(val_errors)]
print(f"Nombre optimal de voisins : {optimal_k}")


knn_optimal = KNeighborsRegressor(n_neighbors=optimal_k)
knn_optimal.fit(X_train, y_train)
y_pred_knn = knn_optimal.predict(X_test)


plt.figure()
plt.scatter(y_test, y_pred_knn, label='Prédictions KNN')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', label='Droite identité')
plt.xlabel('Valeurs réelles')
plt.ylabel('Valeurs prédites')
plt.legend()
plt.title('Prédictions KNN vs Valeurs réelles')
plt.show()

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

mse_knn = mean_squared_error(y_test, y_pred_knn)
mse_lr = mean_squared_error(y_test, y_pred_lr)

print(f"Erreur quadratique moyenne (KNN) : {mse_knn}")
print(f"Erreur quadratique moyenne (Régression linéaire) : {mse_lr}")

if mse_knn < mse_lr:
    print("Le modèle KNN est préférable en termes d'erreur moyenne.")
else:
    print("La régression linéaire est préférable en termes d'erreur moyenne.")



