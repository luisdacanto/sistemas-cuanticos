from scipy import sparse
import numpy as np
import matplotlib.pyplot as plt

#Definiciones

H = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]])
H_sparse = sparse.csr_matrix(H)

def coin(t): 
     return sparse.kron(sparse.identity(2*t+1),H_sparse)

def shift(t):
    forward = np.pad(np.eye(2*t), ((1, 0), (0, 1)))  # paso adelante
    backward = np.pad(np.eye(2*t), ((0, 1), (1, 0)))  # paso atras

    return (
        sparse.kron(forward, np.array([[1, 0], [0, 0]])) +
        sparse.kron(backward, np.array([[0, 0], [0, 1]]))
    )
def evolution_step(t):
    if t==0:
        return np.eye(6)
    return shift(t) @ coin(t)

#Caminata Cuántica
    #psi=np.kron(np.array([0,1,0]),np.array([1,-1j])/np.sqrt(2))
def Psi(t):
    psi=np.kron(np.array([0,1,0]),np.array([1,0j])) #estado inicial
    for i in range (1,t+1):
        psi=evolution_step(i) @ psi
        psi=np.pad(psi,(2,2))
    return psi
#print(Psi(5))

    #Valor esperado de la posición
def prob(t):
    c0=Psi(t)[0::2]
    c1=Psi(t)[1::2]
    return np.abs(c0)**2+ np.abs(c1)**2
def positions(t):
    N=len(prob(t))
    return np.arange(-N//2,N//2)
def expectx(t):
    return np.sum(positions(t)*prob(t))

        #Desviación estándar
def estandar(t):
    var=np.sum((positions(t)**2)*prob(t))-expectx(t)**2
    return np.sqrt(var)

#Gráficas 
    #Distribución de Probabilidad
def plot_prob(t):
    pos = positions(t)
    p = prob(t)
    plt.figure(figsize=(8, 4))
    plt.plot(pos, p, color='blue') 
    plt.title(f"Quantum Walk Probability Distribution at t = {t}")
    plt.xlabel("Position")
    plt.ylabel("Probability")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
plot_prob(100)

#Parámnetros de nuestra caminata
steps=50
longw=np.arange(0,steps)

    #Valor esperado 
E=[expectx(t) for t in longw]
plt.plot(longw,E, 'x', markeredgewidth=2)
plt.xlabel("Tiempo")
plt.ylabel("Valor esperado")
plt.title("Valor esperado de la posición a lo largo del tiempo")
plt.show()

    #Desviación estandar
E=[estandar(t) for t in longw]
plt.plot(longw,E, 'x', markeredgewidth=2)
plt.xlabel("Tiempo")
plt.ylabel("Desviación Estandar")
plt.title("Desviación Estandar de la posición a lo largo del tiempo")
plt.show()