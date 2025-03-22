from scipy import sparse
import numpy as np

#Definiciones

H = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]])
H_sparse = sparse.csr_matrix(H)

def coin(t):
    sparse.kron(sparse.identity(t),H_sparse)

def shift(t):
    forward = np.pad(np.eye(t), ((1, 0), (0, 1)))  # paso adelante
    backward = np.pad(np.eye(t), ((0, 1), (1, 0)))  # paso atras

    return (
        sparse.kron(forward, np.array([[1, 0], [0, 0]])) +
        sparse.kron(backward, np.array([[0, 0], [0, 1]]))
    )
S=shift(3)
print(S.toarray())

def evolution(t):
    return shift(t)*coin(t)

#Cálculos


