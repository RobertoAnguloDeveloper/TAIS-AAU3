import requests
import pandas as pd
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import joblib

# Hacer la solicitud HTTP
url = "http://168.75.79.164:8080/api/parent/all"
response = requests.get(url)
data = response.json()

# Crear DataFrame
df = pd.DataFrame(data)

# Convertir la columna 'record_date' a datetime
df['record_date'] = pd.to_datetime(df['record_date'], format='%d/%m/%Y %H:%M:%S')

# Contar parents por fecha
parents_by_date = df.groupby('record_date').size().reset_index(name='count')

# Preprocesar datos para regresión lineal
parents_by_date['week_of_year'] = parents_by_date['record_date'].dt.isocalendar().week

# Crear y entrenar el modelo de regresión lineal
X = parents_by_date[['week_of_year']].values
y = parents_by_date['count'].values

model = LinearRegression()
model.fit(X, y)

# Guardar el modelo en un archivo
joblib.dump(model, 'linear_regression_model.joblib')

print("Modelo entrenado y guardado exitosamente.")
