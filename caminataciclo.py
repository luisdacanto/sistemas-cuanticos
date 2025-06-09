from scipy import sparse
import numpy as np
import matplotlib.pyplot as plt
from numpy import random

#Definiciones
n_q=6 #número de qubits
L=2**n_q #Largo de nuestro ciclo

H = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]])
H_sparse = sparse.csr_matrix(H)

def coin(L): 
     return sparse.kron(sparse.identity(L),H_sparse)

def shift(L):
    forward = np.roll(np.eye(L),1,1)  # paso adelante
    backward = np.roll(np.eye(L),-1,1)  # paso atras
    
    return (
        sparse.kron(forward, np.array([[1, 0], [0, 0]])) +
        sparse.kron(backward, np.array([[0, 0], [0, 1]])))

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
"Swap de qubits"
def permutar_vector(vector: list, i: int, j: int, width: int) -> list: 
    #Nos da un nuevo vector bajo la permutación de los (qu)bits i y j
    def swap_bits(n: int) -> int:
       #función interna que nos da la imagen de un número "n" bajo la permutación de bits
        pos_i = width - 1 - i
        pos_j = width - 1 - j
        b_i = (n >> pos_i) & 1
        b_j = (n >> pos_j) & 1
        if b_i == b_j:
            return n
        mask = (1 << pos_i) | (1 << pos_j)
        return n ^ mask

    order = [swap_bits(n) for n in range(2**width)]  # Genera el orden de permutación
    return [vector[index] for index in order]

#Probabilidades 
def prob(t,P):
    c0=Psi(t)[0::2]
    c1=Psi(t)[1::2]
    p_original = np.abs(c0)**2 + np.abs(c1)**2

    psi_sw = np.array(permutar_vector(p_original, 0, 1, n_q), dtype=complex)
    c0_sw = psi_sw[0::2]
    c1_sw = psi_sw[1::2]
    probs_sw = np.array(permutar_vector(p_original, 0, 1, n_q), dtype=float) #np.abs(c0_sw)**2 + np.abs(c1_sw)**2

    return (1-P)*p_original+(P)*probs_sw

#Distribución de Probabilidad
def plot_prob(t, P):
    pos = np.arange(L)
    p = prob(t, P)
    plt.plot(pos, p, label=f"P = {P:.2f}")

# Loop over time steps and randomness values
for t in [0,1,30,60,100]:
    plt.figure(figsize=(8, 4))
    for P in [0,0.5,1]:
        plot_prob(t, P)
    
    plt.xlabel("Posición")
    plt.ylabel("Probabilidad")
    plt.title(f"Distribución de Probilidad de la Caminata Cuántica al tiempo t = {t} para varias P")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()




