

from procesamiento import cargar_imagen
import pickle
import numpy as np

with open("entrenamiento_final.pkl", "rb") as f:
    red = pickle.load(f)

ruta = r"D:\perceptron\test\imagen_prueba7.jpeg"
vector = cargar_imagen(ruta)

#probabilidad = red.forward(np.array([vector]))[0][0]
probabilidad = float(red.predict(vector))


umbral = 0.8
if probabilidad >= umbral:
    print(f"00000 Soy yo 00000 (probabilidad: {probabilidad*100:.2f}%)")
elif probabilidad <0.8 and probabilidad >= 0.5:
    print(f"00000 Puedo ser yo 00000 (probabilidad: {probabilidad*100:.2f}%)")
else:
    print(f"XXXXX no soy yo XXXXX (probabilidad: {probabilidad*100:.2f}%)")