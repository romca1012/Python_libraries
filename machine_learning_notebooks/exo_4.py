import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

data_path = './data/breast_cancer_wisconsin.csv'
data = pd.read_csv(data_path)

data.drop(['id'], axis=1, inplace=True)
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})  # Encoder la cible (1 pour Malin, 0 pour Bénin)


X = data.drop('diagnosis', axis=1)
y = data['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


kf = KFold(n_splits=5, shuffle=True, random_state=42)
max_depths = range(1, 21)
train_errors = []
val_errors = []
best_depth = None
best_score = 0

for depth in max_depths:
    clf = DecisionTreeClassifier(max_depth=depth, criterion='gini', random_state=42)
    train_score = []
    val_score = []
    for train_idx, val_idx in kf.split(X_train):
        X_t, X_v = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_t, y_v = y_train.iloc[train_idx], y_train.iloc[val_idx]
        clf.fit(X_t, y_t)
        train_score.append(1 - accuracy_score(y_t, clf.predict(X_t)))
        val_score.append(1 - accuracy_score(y_v, clf.predict(X_v)))
    train_errors.append(np.mean(train_score))
    val_errors.append(np.mean(val_score))
    if np.mean(val_score) < best_score or best_depth is None:
        best_score = np.mean(val_score)
        best_depth = depth

print(f"Profondeur optimale : {best_depth}")
print(f"Erreur de validation minimale : {best_score}")


plt.figure()
plt.plot(max_depths, train_errors, label="Erreur d'entraînement")
plt.plot(max_depths, val_errors, label="Erreur de validation")
plt.xlabel("Profondeur de l'arbre")
plt.ylabel("Erreur")
plt.title("Erreur d'entraînement et de validation en fonction de la profondeur")
plt.legend()
plt.show()


final_model = DecisionTreeClassifier(max_depth=best_depth, criterion='gini', random_state=42)
final_model.fit(X_train, y_train)


plt.figure(figsize=(20, 10))
plot_tree(final_model, feature_names=X.columns, class_names=['B', 'M'], filled=True)
plt.title("Arbre de décision optimisé")
plt.show()


y_pred = final_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Précision : {accuracy}")
print(f"Rappel : {recall}")
print(f"F1 Score : {f1}")


conf_matrix = confusion_matrix(y_test, y_pred)
misclassified = conf_matrix[0, 1] + conf_matrix[1, 0]
print(f"Nombre d'individus mal classés : {misclassified}")


importance = final_model.feature_importances_
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importance}).sort_values(by='Importance', ascending=False)


plt.figure()
plt.bar(importance_df['Feature'], importance_df['Importance'])
plt.xticks(rotation=90)
plt.title("Importance des paramètres pour le diagnostic")
plt.xlabel("Paramètres")
plt.ylabel("Importance")
plt.show()


tree_rules = export_text(final_model, feature_names=list(X.columns))
print(tree_rules)


feature_columns = list(X.columns)

def decision_rule(*args):
    data = dict(zip(feature_columns, args))
    if data['concave points_mean'] <= 0.05:
        if data['radius_worst'] <= 16.83:
            if data['area_se'] <= 48.70:
                if data['smoothness_worst'] <= 0.18:
                    return 0  # Bénin
                else:
                    return 1  # Malin
            else:
                if data['compactness_se'] <= 0.01:
                    return 1  # Malin
                else:
                    return 0  # Bénin
        else:
            if data['texture_mean'] <= 16.19:
                return 0  # Bénin
            else:
                if data['concave points_se'] <= 0.01:
                    return 1  # Malin
                else:
                    return 0  # Bénin
    else:
        if data['concave points_worst'] <= 0.15:
            if data['perimeter_worst'] <= 115.25:
                if data['texture_worst'] <= 27.43:
                    return 0  # Bénin
                else:
                    return 1  # Malin
            else:
                return 1  # Malin
        else:
            if data['fractal_dimension_se'] <= 0.01:
                return 1  # Malin
            else:
                return 0  # Bénin


X_test_subset = X_test[feature_columns].head(10)
manual_predictions = X_test_subset.apply(lambda row: decision_rule(*row), axis=1)
sklearn_predictions = final_model.predict(X_test_subset)


comparison_df = pd.DataFrame({
    'Vrai diagnostic': y_test.head(10).values,
    'Prédiction sklearn': sklearn_predictions,
    'Prédiction manuelle': manual_predictions
})
print(comparison_df)


incoherences = comparison_df[comparison_df['Prédiction sklearn'] != comparison_df['Prédiction manuelle']]
print(f"Incohérences détectées : {len(incoherences)}")