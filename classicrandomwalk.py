import numpy as np
import matplotlib.pyplot as plt
import random

n_p=1000
n_c=10000
posiciones_finales=[]

for i in range(n_c):
    posicion= 0
    for _ in range(n_p):
        movimiento = random.choice([-1,1])
        posicion+=movimiento
    posiciones_finales.append(posicion)

plt.hist(posiciones_finales, bins=50, density=True, alpha=0.6, color='b')
plt.title('Distribución de la Caminata Aleatoria')
plt.xlabel('Posición')
plt.ylabel('Densidad de Probabilidad')
plt.grid(True)
plt.show()