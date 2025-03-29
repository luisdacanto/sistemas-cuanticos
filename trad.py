from scipy import sparse
import numpy as np

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
    return shift(t) @ coin(t)

#Caminata Cuántica
psi=np.kron(np.array([0,1,0]),np.array([1,0]))

def Psi(t):
    for i in range (1,t+1):
        psi=evolution_step(i) @ psi
        psi=np.pad(psi,(2,2))
    return psi

#Valor esperado de la posición

def x(t):
    return 