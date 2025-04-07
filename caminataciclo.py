from scipy import sparse
import numpy as np
import matplotlib.pyplot as plt
from numpy import random

#Definiciones

L= 64  #Largo de nuestro ciclo

H = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]])
H_sparse = sparse.csr_matrix(H)

def coin(L): 
     return sparse.kron(sparse.identity(L),H_sparse)

def shift(L):
    forward = np.pad(np.eye(L-1), ((1, 0), (0, 1)))  # paso adelante
    backward = np.pad(np.eye(L-1), ((0, 1), (1, 0)))  # paso atras
    
    ket_0 = np.zeros(L)
    ket_0[0]=1

    ket_L = np.zeros(L)
    ket_L[L-1]=1

    zero_to_L=sparse.kron(np.outer(ket_0,ket_L), np.array([[1, 0], [0, 0]]))
    L_to_zero=sparse.kron(np.outer(ket_L,ket_0),np.array([[0, 0], [0, 1]]))
    return (
        sparse.kron(forward, np.array([[1, 0], [0, 0]])) +
        sparse.kron(backward, np.array([[0, 0], [0, 1]])) +
        zero_to_L + L_to_zero
    )

def evolution_step(L):
     return shift(L)@ coin(L)

def Psi(t): 
    inicial=np.zeros(L)
    inicial[0]=1
    psi=np.kron(inicial,np.array([1,0j])) #estado inicial
    for i in range (1,t+1):
        psi=evolution_step(L) @ psi
    return psi

#Compuertas Switch
def S(vec, i):
    vec = vec.copy()
    vec[i], vec[i + 1] = vec[i + 1], vec[i]
    return vec

def S_Psi(t):
    return S(Psi(t), random.randint(L - 1)) 

#Probabilidades 
def prob(t):
    c0=Psi(t)[0::2]
    c1=Psi(t)[1::2]
    return np.abs(c0)**2+ np.abs(c1)**2

#Distribución de Probabilidad
def plot_prob(t):
    pos = np.arange(L)
    p = prob(t)
    plt.figure(figsize=(8, 4))
    plt.plot(pos, p, color='blue') 
    plt.title(f"Quantum Walk Probability Distribution at t = {t}")
    plt.xlabel("Position")
    plt.ylabel("Probability")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
plot_prob(60)

