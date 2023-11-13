import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.exceptions import UndefinedMetricWarning
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import warnings

# Desactivar las advertencias de métricas indefinidas
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# 1. Cargar el conjunto de datos
data = pd.read_csv('winequality-white.csv', sep=';')

# 2. Preprocesamiento de datos
X = data.drop('quality', axis=1)
y = data['quality']

# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Modelos de clasificación
# 3.1 Regresión Logística
logistic_model = LogisticRegression(max_iter=1000, random_state=42)
logistic_model.fit(X_train_scaled, y_train)
y_pred_logistic = logistic_model.predict(X_test_scaled)
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)

# 3.2 K Vecinos más Cercanos (KNN)
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
y_pred_knn = knn_model.predict(X_test_scaled)
accuracy_knn = accuracy_score(y_test, y_pred_knn)

# 3.3 Máquinas de Soporte Vectorial (SVM)
svm_model = SVC(kernel='linear', C=1)
svm_model.fit(X_train_scaled, y_train)
y_pred_svm = svm_model.predict(X_test_scaled)
accuracy_svm = accuracy_score(y_test, y_pred_svm)

# 4. Red Neuronal con TensorFlow/Keras
# Cambio en la capa de salida y codificación de etiquetas
num_classes = len(np.unique(y_train))
nn_model = Sequential([
    Dense(64, activation='relu', input_dim=X_train.shape[1]),
    Dense(num_classes, activation='softmax')  # Cambio en la capa de salida
])
nn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Codificación de etiquetas
y_train_encoded = to_categorical(y_train - 3, num_classes)  # Resta 3 para ajustar los índices a partir de 0
y_test_encoded = to_categorical(y_test - 3, num_classes)

# Entrenamiento del modelo
nn_model.fit(X_train_scaled, y_train_encoded, epochs=10, batch_size=32, validation_data=(X_test_scaled, y_test_encoded))

# Predicción y evaluación
y_pred_nn = np.argmax(nn_model.predict(X_test_scaled), axis=1) + 3  # Suma 3 para ajustar los índices de nuevo
accuracy_nn = accuracy_score(y_test, y_pred_nn)

# Imprimir resultados
print(f'Precisión de Regresión Logística: {accuracy_logistic}')
print(f'Precisión de KNN: {accuracy_knn}')
print(f'Precisión de SVM: {accuracy_svm}')
print(f'Precisión de la Red Neuronal: {accuracy_nn}')

# Informe de clasificación para la Regresión Logística
print("\nInforme de Clasificación (Regresión Logística):")
print(classification_report(y_test, y_pred_logistic))

# Informe de clasificación para KNN
print("\nInforme de Clasificación (KNN):")
print(classification_report(y_test, y_pred_knn))

# Informe de clasificación para SVM
print("\nInforme de Clasificación (SVM):")
print(classification_report(y_test, y_pred_svm))

# Informe de clasificación para la Red Neuronal
print("\nInforme de Clasificación (Red Neuronal):")
print(classification_report(y_test, y_pred_nn))
