# Redes neuronales con las condiciones iniciales

## tanh_buena_v2
Esta es la red con la mejor precisión, validada en el espacio de parámetros mostrado en results_NN.ipynb

## ejemplo_de_uso.ipynb
Este archivo tiene instrucciones de como generar un par de condiciones iniciales para dar evolución a la ecuación de perturbaciones de materia. Está guiado y lleva a los solvers que usé en la tesis.

## red_kfold_cv

Es una red entrenada con el método de kfold para dividir los datos (https://github.com/PedroRozin/Tesis2025/blob/main/python/grilla_NN_2_kfold.py). Fue hecho solamente para verificar que la división original de datos para entrenamiento era efectivamente representativa del conjunto total. Da todo ok.