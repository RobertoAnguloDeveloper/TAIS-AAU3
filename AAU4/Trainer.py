import requests
import pandas as pd
from sklearn.linear_model import ElasticNet
from datetime import datetime, timedelta
import joblib
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np

# Obtener datos de estudiantes
url_students = "http://168.75.79.164:8080/api/student/all"
response_students = requests.get(url_students)
data_students = response_students.json()

# Obtener datos de padres
url_parents = "http://168.75.79.164:8080/api/parent/all"
response_parents = requests.get(url_parents)
data_parents = response_parents.json()

# Crear DataFrames para estudiantes y padres
df_students = pd.DataFrame(data_students)
df_parents = pd.DataFrame(data_parents)

# Normalizar el campo 'parent' en el DataFrame de estudiantes
df_students_normalized = pd.json_normalize(df_students['parent'])

# Fusionar DataFrames usando la información del ID del padre
df_merged = pd.merge(df_students, df_students_normalized, left_index=True, right_index=True, suffixes=('_student', '_parent'))

# Contar familias únicas
unique_families = df_merged[['id_parent', 'firstName_parent', 'lastName_parent']].drop_duplicates().shape[0]

# Mostrar el número de familias
print(f"Número de familias registradas: {unique_families}")

# Preprocesar datos para regresión lineal
df_merged['record_date'] = pd.to_datetime(df_merged['record_date'], format='%d/%m/%Y %H:%M:%S')
parents_by_date = df_merged.groupby('record_date').size().reset_index(name='count')

# Crear la columna 'week_of_year'
parents_by_date['week_of_year'] = parents_by_date['record_date'].dt.isocalendar().week

# Crear y entrenar el modelo ElasticNet
X = parents_by_date[['week_of_year']].values
y = parents_by_date['count'].values

model = ElasticNet(alpha=0.5, l1_ratio=0.05)
model.fit(X, y)

# Guardar el modelo en un archivo
joblib.dump(model, 'elasticnet_model.joblib')

# Mostrar el número de familias predicho para la próxima semana
start_date = datetime.now() + timedelta(days=7)
next_week = start_date.isocalendar().week
X_prediction = pd.DataFrame({'week_of_year': [next_week]})
prediction = model.predict(X_prediction)
print(f"Número de familias predicho para la próxima semana: {int(prediction[0])}")

# Crear la curva de aprendizaje
train_sizes, train_scores, test_scores = learning_curve(
    model, X, y, cv=5, scoring='neg_mean_squared_error', train_sizes=np.linspace(0.1, 1.0, 50))

# Calcular los valores promedio y las desviaciones estándar de los puntajes
train_scores_mean = -np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = -np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# Graficar la curva de aprendizaje
plt.figure(figsize=(10, 6))
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Entrenamiento")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Validación cruzada")
plt.xlabel('Tamaño del conjunto de entrenamiento')
plt.ylabel('Error cuadrático medio')
plt.title('Curva de Aprendizaje de ElasticNet')
plt.legend(loc="best")
plt.show()
