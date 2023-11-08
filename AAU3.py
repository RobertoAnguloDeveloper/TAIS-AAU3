import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Cargar la muestra de datos
data = pd.read_csv("Dataset.csv", parse_dates=["Posted On"], sep=",")

# Gráfico de BHK vs. Rent
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.scatter(data["BHK"], data["Rent"])
plt.xlabel("Número de habitaciones (BHK)")
plt.ylabel("Precio del alquiler (Rent)")
plt.title("BHK vs. Rent")

# Gráfico de Size vs. Rent
plt.subplot(1, 3, 2)
plt.scatter(data["Size"], data["Rent"])
plt.xlabel("Tamaño (Size)")
plt.ylabel("Precio del alquiler (Rent)")
plt.title("Size vs. Rent")

# Gráfico de Furnishing Status vs. Rent
plt.subplot(1, 3, 3)
furnishing_groups = data.groupby("Furnishing Status")["Rent"].mean()
furnishing_groups.plot(kind="bar")
plt.xlabel("Estado de amueblamiento")
plt.ylabel("Precio promedio del alquiler (Rent)")
plt.title("Furnishing Status vs. Rent")

# Filtrar y limpiar los datos
data = data[["BHK", "Rent"]]
data.dropna(inplace=True)

# Definir la variable independiente (X) y la variable dependiente (Y)
x = data["BHK"]
y = data["Rent"]

# Agregar una constante (intercepto) a la variable independiente
x = sm.add_constant(x)

# Ajustar el modelo de regresión lineal
model = sm.OLS(y, x).fit()

# Mostrar la tabla de regresión
print(model.summary())

plt.tight_layout()
plt.show()
