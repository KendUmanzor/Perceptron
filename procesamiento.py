import numpy as np
from PIL import Image
import os

def cargar_imagen(ruta, tamano=(100, 100)):
    img = Image.open(ruta).convert('L').resize(tamano)
    return np.array(img).flatten() / 255.0

def cargar_carpeta(directorio, tamano=(100, 100)):
    X = []
    y = []
    for etiqueta, clase in enumerate(["otros", "yo"]):
        clase_dir = os.path.join(directorio, clase)
        if not os.path.isdir(clase_dir):
            continue
        for archivo in os.listdir(clase_dir):
            ruta = os.path.join(clase_dir, archivo)
            try:
                vector = cargar_imagen(ruta, tamano)
                X.append(vector)
                y.append(etiqueta)
            except:
                print(f"Error cargando la imagen: {ruta}")
    return np.array(X), np.array(y)
