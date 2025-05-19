import pandas as pd
import matplotlib.pyplot as plt
from pycaret.regression import *

# Charger les données
data_path = './data/CO2_Emissions_Canada.csv'
data = pd.read_csv(data_path)

# Prétraitement des données
# Extraction des 20 dernières lignes pour prédictions ultérieures
data_CO2_emission_pred = data.tail(20).reset_index(drop=True)
data = data.iloc[:-20].reset_index(drop=True)  # Supprimer les 20 dernières lignes pour l'entraînement

# Initialiser PyCaret
setup(
    data=data,
    target='CO2 Emissions(g/km)',
    train_size=0.7,  # Utiliser 70% des données pour l'entraînement
    normalize=True,  # Appliquer une normalisation centrée réduite
    categorical_features=['Make', 'Model', 'Vehicle Class', 'Transmission', 'Fuel Type'],  # Variables qualitatives
    numeric_features=['Engine Size(L)', 'Cylinders', 'Fuel Consumption City (L/100 km)', 
                    'Fuel Consumption Hwy (L/100 km)', 'Fuel Consumption Comb (L/100 km)', 
                    'Fuel Consumption Comb (mpg)'],  # Variables numériques
    fold=5,  # Utiliser 5-fold pour la validation croisée
    session_id=42  # Assurer la reproductibilité
)

# Entraîner un modèle de type Random Forest
rf_model = create_model('rf')

# Optimiser le modèle
optimized_rf_model = tune_model(rf_model, n_iter=10)  # Optimisation des hyperparamètres

# Afficher les caractéristiques les plus importantes
print("\nCaractéristiques les plus importantes :")
plot_model(optimized_rf_model, plot='feature')

# Graphique de validation (MSE en fonction de l'itération d'optimisation)
tuning_results = pull()  # Récupérer les résultats d'optimisation
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(tuning_results) + 1), tuning_results['MSE'], marker='o', label='MSE')
plt.xlabel('Itérations d\'optimisation')
plt.ylabel('Erreur quadratique moyenne (MSE)')
plt.title('MSE en fonction des itérations d\'optimisation')
plt.legend()
plt.grid()
plt.show()

# Évaluer le modèle optimisé
evaluate_model(optimized_rf_model)

# Effectuer des prédictions sur les données extraites (data_CO2_emission_pred)
data_CO2_emission_pred_encoded = predict_model(optimized_rf_model, data=data_CO2_emission_pred)

# Afficher les prédictions
print("\nPrédictions sur les 20 dernières lignes :")
print(data_CO2_emission_pred_encoded[['Make', 'Model', 'CO2 Emissions(g/km)', 'Label']])  # Label = prédictions

