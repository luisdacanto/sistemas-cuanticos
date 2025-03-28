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
    return shift(t)@coin(t)

#Cálculos
psi=np.pad(np.array([1, 1]) / np.sqrt(2),2) #estado inicial

#Caminata Cuántica

for t in range(7):
    psi=evolution_step(t).dot(psi)
    psi=np.pad(psi,(2,2))

print(psi)