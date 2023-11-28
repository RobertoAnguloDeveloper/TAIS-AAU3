import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("Dataset.csv")

data = data[["Rent", "Size"]]
data.dropna(inplace=True)

entrada = np.array(data[["Rent"]], dtype=float)
salida = np.array(data[["Size"]], dtype=float)

capa1 = tf.keras.layers.Dense(units=1, input_shape=[1])

modelo = tf.keras.Sequential([capa1])

modelo.compile(
    optimizer = tf.keras.optimizers.Adam(0.1),
    loss = "mean_squared_error"
)

print("Entrenando la red")

entrenamiento = modelo.fit(entrada, salida, epochs=2500, verbose=False)

modelo.save('RedNeuronal.keras')
modelo.save_weights('Weights.keras')

plt.xlabel("Ciclos de entrenamiento")
plt.ylabel("Errores")

plt.plot(entrenamiento.history["loss"])

print("Terminamos")

i = input("Ingresa el numero: ")
i = float(i)

prediction = modelo.predict([i])

print("La prediccion es: " + str(prediction))
plt.show()