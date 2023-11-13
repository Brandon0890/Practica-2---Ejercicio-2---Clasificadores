import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error

# Cargar el dataset desde el archivo de texto, considerando que las columnas están separadas por tabulaciones
data = pd.read_csv('AutoInsurSweden.txt', delimiter='\t')

# Explorar el dataset
print("Primeras filas del dataset:")
print(data.head())

# Convertir la columna 'Y' a números de punto flotante
data['Y'] = data['Y'].str.replace(',', '.').astype(float)

# Separar características (X) y etiquetas (Y)
X = data[['X']]  # Asumiendo que 'X' es la columna de número de reclamaciones
Y = data[['Y']]  # Asumiendo que 'Y' es la columna de pagos totales en miles de coronas suecas

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Escalar características para algunos algoritmos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Implementar Regresión Lineal
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, Y_train)
y_pred_linear = linear_model.predict(X_test_scaled)
mse_linear = mean_squared_error(Y_test, y_pred_linear)
print(f'Mean Squared Error (Regresión Lineal): {mse_linear}')

# Implementar K-Vecinos más Cercanos para Regresión
knn_model = KNeighborsRegressor()
knn_model.fit(X_train_scaled, Y_train)
y_pred_knn = knn_model.predict(X_test_scaled)
mse_knn = mean_squared_error(Y_test, y_pred_knn)
print(f'Mean Squared Error (K-Vecinos más Cercanos para Regresión): {mse_knn}')

# Implementar una red neuronal con TensorFlow para Regresión
neural_network = Sequential()
neural_network.add(Dense(units=64, activation='relu', input_dim=1))  # Ajusta input_dim según el número de características
neural_network.add(Dense(units=1, activation='linear'))  # Linear activation para regresión
neural_network.compile(optimizer='adam', loss='mean_squared_error')  # Usamos MSE para regresión
neural_network.fit(X_train_scaled, Y_train, epochs=10, batch_size=32, validation_data=(X_test_scaled, Y_test))
y_pred_nn = neural_network.predict(X_test_scaled)
mse_nn = mean_squared_error(Y_test, y_pred_nn)
print(f'Mean Squared Error (Red Neuronal para Regresión): {mse_nn}')
