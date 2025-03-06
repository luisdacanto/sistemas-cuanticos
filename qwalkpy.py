import numpy as np
import matplotlib.pyplot as plt
from functools import lru_cache 

#Definamos nuestras funciones de  amplitud de probabilidad

@lru_cache(None)
def A(n,t):
    if t==0:
        return 1 if n==0 else 0
    return (A(n-1,t-1)+B(n-1,t-1))/np.sqrt(2)

@lru_cache(None)
def B(n,t):
    if t==0:
        return 0
    return (A(n+1,t-1)-B(n+1,t-1))/np.sqrt(2)

#Calculamos la probabilidad de encontrar al caminante en una posición n dado t pasos

def p(n,t):
    return np.absolute(A(n,t))**2+np.absolute(B(n,t))**2

#Parámnetros de nuestra caminata
steps=100
longw=np.arange(-(steps+1),steps+1)
prob=[p(n,steps) for n in longw] 

plt.plot(longw,prob, color="blue")
plt.xlabel("Position")
plt.ylabel("Probability")
plt.title("Quantum Walk")

plt.show()