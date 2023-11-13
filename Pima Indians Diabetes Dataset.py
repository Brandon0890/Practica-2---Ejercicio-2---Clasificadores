import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Cargar el conjunto de datos
url = "D:\Tareas_Varias_py\pima-indians-diabetes.csv"  # Reemplazar con la URL real o la ruta del archivo
df = pd.read_csv(url, header=None)

# Dividir el conjunto de datos en características (X) y etiquetas (y)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar las características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Regresión Logística
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)
logistic_predictions = logistic_model.predict(X_test)
logistic_accuracy = accuracy_score(y_test, logistic_predictions)
print(f"Regresión Logística Accuracy: {logistic_accuracy}")

# K-Vecinos Cercanos
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)
knn_predictions = knn_model.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_predictions)
print(f"K-Vecinos Cercanos Accuracy: {knn_accuracy}")

# Máquinas de Soporte Vectorial
svm_model = SVC()
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_predictions)
print(f"Máquinas de Soporte Vectorial Accuracy: {svm_accuracy}")

# Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_predictions = nb_model.predict(X_test)
nb_accuracy = accuracy_score(y_test, nb_predictions)
print(f"Naive Bayes Accuracy: {nb_accuracy}")

# Red Neuronal con ajustes
nn_model = MLPClassifier(max_iter=5000, learning_rate='constant', hidden_layer_sizes=(100,), alpha=0.0001, random_state=42)
nn_model.fit(X_train, y_train)
nn_predictions = nn_model.predict(X_test)
nn_accuracy = accuracy_score(y_test, nn_predictions)
print(f"Red Neuronal Accuracy: {nn_accuracy}")

# Validación Cruzada
cross_val_scores = cross_val_score(nn_model, X, y, cv=5)
print(f"Cross-validated Accuracy: {cross_val_scores.mean()}")
