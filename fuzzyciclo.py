from scipy import sparse
import numpy as np
import matplotlib.pyplot as plt
from numpy import random

# Parámetros
n_q=6 #número de qubits
L = 2**n_q   # número de posiciones
H = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]])  # Hadamard
H_sp = sparse.csr_matrix(H)
I_L = sparse.identity(L, format='csr')

#Función de intercambiar qubits
def Swap_q(n, i, j, width=None):
    # Determinar ancho de la representación
    if width is None:
        # n.bit_length() da 0 si n == 0, así que garantizamos al menos 1
        width = max(n.bit_length(), 1)

    # Calculamos las posiciones desde el LSB
    pos_i = width - 1 - i
    pos_j = width - 1 - j

    # Extraemos esos bits
    bit_i = (n >> pos_i) & 1
    bit_j = (n >> pos_j) & 1

    # Si son iguales, no hay nada que hacer
    if bit_i == bit_j:
        return n

    # Máscara con 1s en pos_i y pos_j para hacer XOR y flip
    mask = (1 << pos_i) | (1 << pos_j)
    return n ^ mask


# 1) Pre–computar operador moneda y operador desplazamiento
coin_op   = sparse.kron(I_L, H_sp, format='csr')

# Construimos la matriz de desplazamiento con condiciones periódicas usando roll
forward  = sparse.csr_matrix(np.roll(np.eye(L),  1, axis=0))
backward = sparse.csr_matrix(np.roll(np.eye(L), -1, axis=0))
C0 = sparse.csr_matrix([[1,0],[0,0]])
C1 = sparse.csr_matrix([[0,0],[0,1]])

shift_op = sparse.kron(forward,  C0, format='csr') + \
           sparse.kron(backward, C1, format='csr')

# 2) Pre–computar el único evolution_step U = shift ∘ coin
U = shift_op.dot(coin_op)   # U es dispersa de tamaño (2L)x(2L)

# Función que devuelve psi tras t pasos
def Psi(t):
    psi = np.zeros(2*L, dtype=complex)
    psi[0] = 1  # coin=0 ⊗ pos=0
    for _ in range(t):
        psi = U.dot(psi)
    return psi

# 3) Distribución de probabilidad, vectorizada
def prob(t, P):
    psi = Psi(t)
    c0 = psi[0::2]
    c1 = psi[1::2]
    p = np.abs(c0)**2 + np.abs(c1)**2

    if P == 0:
        return p

    # intercambio aleatorio de una posición
    k = random.randint(L)
    p_swapped = np.roll(p, 1)
    return (1 - P)*p + P*p_swapped

def plot_prob(t, P):
    plt.plot(np.arange(L), prob(t, P), label=f"P={P:.1f}")