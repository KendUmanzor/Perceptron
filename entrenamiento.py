

    
from Red_neuronal import RedNeuronal
from procesamiento import cargar_carpeta
import pickle

X, y = cargar_carpeta(r"D:\perceptron\datos")

# Red neuronal: 400 entradas, 16 neuronas ocultas, 1 salida
#red = RedNeuronal(entradas=10000, ocultas=80, salidas=1)


red= RedNeuronal([10000, 512, 128, 1],activation='sigmoid')
red.entrenar(X,y,learning_rate=0.05, epochs=1000)
#red.entrenar(X, y, epochs=1000, lr=0.5)

with open("entrenamiento_final1.pkl", "wb") as f:
    pickle.dump(red, f)