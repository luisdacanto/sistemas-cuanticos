import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from numpy import random

"Parámetros de la caminata"
n_q=6 #número de qubits
L=2**n_q #largo de la caminata

"Operadores"
    #Moneda
H=(1/np.sqrt(2)) * np.array([[1, 1], [1, -1]]) #Compuerta de Hadamard
Coin=np.kron(np.identity(L),H)
    #Operador de Paso
foward=np.roll(np.identity(L),1,1) #Paso adelante
backward=np.roll(np.identity(L),-1,1) #Paso atrás
Shift=np.kron(foward,np.array([[1, 0], [0, 0]])) + np.kron(backward,np.array([[0, 0], [0, 1]])) #Operador Shift
    #Operador de Evolución
U=Shift.dot(Coin)

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

"Evolución temporal y Probabilidades"

def Psi(t): #evolución temporal
    p_i=np.zeros(L,dtype=complex)
    p_i[0]=1/np.sqrt(2)
    p_i[L//2]=1/np.sqrt(2) #posición inicial |1>
    c_i=np.array([1,0j]) #estado moneda
    psi=np.kron(p_i,c_i) #estado inicial
    for _ in range (1,t+1):
        psi=U.dot(psi)
    return psi

def prob(t: int, P: float) -> np.ndarray: #Distribución de probabilidad
    psi = Psi(t)
    #Distribución normal
    c0 = psi[0::2]
    c1 = psi[1::2]
    probs = np.abs(c0)**2 + np.abs(c1)**2

    #Distribución después del swap

    psi_sw = np.array(permutar_vector(psi, 0, 1, n_q+1), dtype=complex)
    c0_sw = psi_sw[0::2]
    c1_sw = psi_sw[1::2]
    probs_sw = np.array(permutar_vector(probs, 4, 5, n_q), dtype=float) #np.abs(c0_sw)**2 + np.abs(c1_sw)**2

    return (1.0 - P)*probs + P*probs_sw

def Switch(v,pesos): #Canal Switch
    alpha = [1,1,1,1,1,1]
    Pesos = np.random.dirichlet(alpha)  #vector de probabilidades (las entradas suman 1)
    #Construimos la suma ponderada de permutaciones
    x =pesos[0]* v
    for i in range(1,n_q):
        vp = np.array(permutar_vector(v, i-1,i, n_q))
        x += pesos[i] * vp
    return x

def Prob_fuzzy(t: int, pesos)-> np.ndarray:
    psi = Psi(t)
    #Distribución normal
    c0 = psi[0::2]
    c1 = psi[1::2]
    probs = np.abs(c0)**2 + np.abs(c1)**2

    return Switch(probs, pesos)

"Gráficas"

#Distribución de Probabilidad
def plot_prob_fuzzy(t: int):
    pos = np.arange(L)
    p   = Prob_fuzzy(t)
    plt.plot(pos, p, label=f"t = {t}")

coleccion_pesos = {
    "P=0": [1,    0,    0,    0,    0,    0],
    "P=0.2": [0.8,0.04,0.04,0.04,0.04,0.04],
    "P=0.5": [0.5,0.1,0.1,0.1,0.1,0.1],
    "P=1": [0,0.2,0.2,0.2,0.2,0.2]
}

# Función de plotting comparativo
def plot_comparison(t: int, coleccion_pesos: dict):
    pos = np.arange(L)
    plt.figure(figsize=(8,4))
    for etiqueta, pw in coleccion_pesos.items():
        p = Prob_fuzzy(t, pw)
        plt.plot(pos, p, label=etiqueta)
    plt.xlabel("Posición")
    plt.ylabel("Probabilidad")
    plt.title(f"Distribución de Probilidad de la Caminata Fuzzy al tiempo t = {t} para varias P")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Genera la comparación en varios instantes
for t in [0,1,10,50,100]:
    plot_comparison(t, coleccion_pesos)

for t in [0,1,10,50,100]:
    print(sum(Prob_fuzzy(t,[0,0.2,0.2,0.2,0.2,0.2])))