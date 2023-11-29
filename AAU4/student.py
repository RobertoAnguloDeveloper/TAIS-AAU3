import pandas as pd
from datetime import datetime, timedelta
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests

# Cargar el modelo entrenado
model = joblib.load('elasticnet_model.joblib')

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

# Hacer la predicción para la próxima semana
start_date = datetime.now() + timedelta(days=7)
next_week = start_date.isocalendar().week
next_week_df = pd.DataFrame({'week_of_year': [next_week]})
prediction = model.predict(next_week_df)

# Mostrar el número de familias y la predicción para la próxima semana
print(f"Número de familias registradas: {unique_families}")
print(f"Predicción para la próxima semana: {int(prediction[0])} familias")

# Graficar
fig, ax = plt.subplots()
bars = ax.bar(['Familias registradas', 'Predicción próxima semana'], [unique_families, prediction[0]], color=['blue', 'red'])
ax.set_xlabel('Categorías')
ax.set_ylabel('Cantidad')
ax.set_title('Número de Familias Registradas y Predicción para la Próxima Semana')

# Agregar etiquetas a las barras
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval), ha='center', va='bottom')

plt.show()
