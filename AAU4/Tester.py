import pandas as pd
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests

# Cargar el modelo entrenado
model = joblib.load('linear_regression_model.joblib')

# Hacer la predicción para la próxima semana
start_date = datetime.now() + timedelta(days=7)
next_week = start_date.isocalendar().week
next_week_df = pd.DataFrame({'week_of_year': [next_week]})
prediction = model.predict(next_week_df)

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

# Ajustar el tamaño de la ventana de las gráficas
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

# Gráfico de regresión lineal
ax1.plot(parents_by_date['record_date'], parents_by_date['count'], color='blue', marker='o', label='Datos reales')
ax1.scatter(start_date, prediction, color='red', label=f'Predicción: {int(prediction[0])} parents')
ax1.set_xlabel('Fecha')
ax1.set_ylabel('Cantidad de parents')
ax1.set_title('Regresión Lineal: Datos reales y predicción para la próxima semana')
ax1.legend()
ax1.xaxis.set_major_locator(mdates.MonthLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
ax1.tick_params(axis='x', rotation=45)
ax1.get_xaxis().set_major_locator(plt.MaxNLocator(nbins=8))  # Mostrar hasta 8 fechas en el eje x

# Anotaciones en el gráfico de regresión lineal
for i, txt in enumerate(parents_by_date['count']):
    ax1.annotate(txt, (parents_by_date['record_date'].iloc[i], txt), textcoords="offset points", xytext=(0, 1), ha='center')

# Gráfico de barras
ax2.bar(parents_by_date['record_date'], parents_by_date['count'], color='blue', label='Datos reales')
ax2.bar(start_date, prediction, color='red', label=f'Predicción: {int(prediction[0])} parents')
ax2.set_xlabel('Fecha')
ax2.set_ylabel('Cantidad de parents')
ax2.set_title('Gráfico de Barras: Datos reales y predicción para la próxima semana')
ax2.legend()
ax2.xaxis.set_major_locator(mdates.MonthLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
ax2.tick_params(axis='x', rotation=45)
ax2.get_xaxis().set_major_locator(plt.MaxNLocator(nbins=8))  # Mostrar hasta 8 fechas en el eje x

# Anotaciones en el gráfico de barras
for i, txt in enumerate(parents_by_date['count']):
    ax2.annotate(txt, (parents_by_date['record_date'].iloc[i], txt), textcoords="offset points", xytext=(0, 5), ha='center')

plt.tight_layout(rect=[0, 0.01, 0.5, 0.9])  # Ajustar el espacio en la parte superior y alrededor de la figura
plt.show()
