import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Charger les données
data_path = './data/segmentation_data.csv'
data = pd.read_csv(data_path)

# Définir la fonction de normalisation
def min_max(df):
    scaler = MinMaxScaler()
    return pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# Normalisation des données
data_normalized = min_max(data)

# Déterminer le nombre optimal de clusters avec la méthode de la silhouette
range_clusters = range(2, 8)
silhouette_scores = []

for n_clusters in range_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(data_normalized)
    silhouette_avg = silhouette_score(data_normalized, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Afficher le graphique du score de silhouette
plt.figure()
plt.plot(range_clusters, silhouette_scores, marker='o', color='cyan')
plt.title("Méthode de la silhouette pour déterminer le nombre optimal de clusters")
plt.xlabel("Nombre de clusters")
plt.ylabel("Score de silhouette")
plt.show()
