from scipy import sparse
import numpy as np
import matplotlib.pyplot as plt
from numpy import random

n_q=3 # número de qubits

L=2**n_q #largo de nuestra caminata

def in_to_bin(n,n_q): #pasar de la posición a una lista a binario
    b=bin(n)[2:]
    bi=b.zfill(n_q)
    return np.array([int(d) for d in bi], dtype=int)

def S(vec, i,j): #intercambiar las entradas i,j
    vec = vec.copy()
    vec[i], vec[j] = vec[j], vec[i]
    return vec

def bin_to_in(arr):
    bits=arr.tolist()
    bit_str="".join(str(b) for b in bits)
    return int(bit_str,2)


 #versión eficiente de chatgpt   
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

for i in range(L):
    print (i, "se mapea a", Swap_q(i,0,2,n_q))

k=random.randint(n_q)
l=random.randint(n_q)
    
vector_orden=[Swap_q(n,k,l,n_q) for n in range(2*L)]

print(vector_orden)

def permutar(vector,orden):
        return [vector[i] for i in orden]



# Parámetros globales
n_q = 6                  # número de qubits de posición
L   = 2**n_q             # número de posiciones
width = n_q + 1          # total de bits (1 coin + n_q posiciones)

# Operadores 
H     = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]])
H_sp  = sparse.csr_matrix(H)
I_L   = sparse.identity(L, format='csr')
coin_op   = sparse.kron(I_L, H_sp, format='csr')

forward   = sparse.csr_matrix(np.roll(np.eye(L),  1, axis=0))
backward  = sparse.csr_matrix(np.roll(np.eye(L), -1, axis=0))
C0        = sparse.csr_matrix([[1,0],[0,0]])
C1        = sparse.csr_matrix([[0,0],[0,1]])
shift_op  = sparse.kron(forward,  C0, format='csr') + \
            sparse.kron(backward, C1, format='csr')

U = shift_op.dot(coin_op)   # dimensión (2L)x(2L)


# —————————————————————————
# 1) Función de swap a nivel bit
# —————————————————————————
def swap_index(n: int, i: int, j: int, width: int) -> int:
    """
    Intercambia los bits i y j (0 = MSB) en la representación de n
    usando exactamente 'width' bits, y devuelve el nuevo entero.
    O(1) y sin conversiones a str.
    """
    pos_i = width - 1 - i
    pos_j = width - 1 - j
    # extraemos
    b_i = (n >> pos_i) & 1
    b_j = (n >> pos_j) & 1
    if b_i == b_j:
        return n
    # máscara de flip
    mask = (1 << pos_i) | (1 << pos_j)
    return n ^ mask


# —————————————————————————————
# 2) Swap directo sobre el vector psi
# —————————————————————————————
def swap_state(psi: np.ndarray, i: int, j: int, width: int) -> np.ndarray:
    """
    Dado el estado psi (longitud 2*L), intercambia el bit i con j
    en el índice de cada amplitud, devolviendo el nuevo psi.
    O(N) en tiempo pero sin listas intermedias costosas.
    """
    new_psi = psi.copy()
    N = psi.shape[0]
    for n in range(N):
        m = swap_index(n, i, j, width)
        if m > n:
            # solo swap una vez
            new_psi[n], new_psi[m] = new_psi[m], new_psi[n]
    return new_psi


# —————————————————————————
# 3) Evolución y probabilidad
# —————————————————————————
def Psi(t: int) -> np.ndarray:
    psi = np.zeros(2*L, dtype=complex)
    psi[0] = 1  # estado inicial |coin=0,pos=0>
    for _ in range(t):
        psi = U.dot(psi)
    return psi

def prob(t: int, P: float) -> np.ndarray:
    """
    Devuelve la mezcla (1-P)*p + P*p_swapped, donde p es la
    probabilidad normal y p_swapped la probabilidad de un psi con
    dos bits de su índice intercambiados al azar.
    """
    psi = Psi(t)
    # distribución "sin swap"
    c0 = psi[0::2]
    c1 = psi[1::2]
    p = np.abs(c0)**2 + np.abs(c1)**2

    if P == 0.0:
        return p

    # elige dos qubits al azar de la representación (0=coin, 1..n_q=pos)
    i, j = random.randint(width), random.randint(width)
    psi_sw = swap_state(psi, i, j, width)
    c0_sw = psi_sw[0::2]
    c1_sw = psi_sw[1::2]
    p_sw = np.abs(c0_sw)**2 + np.abs(c1_sw)**2

    return (1.0 - P)*p + P*p_sw

def plot_prob(t: int, P: float):
    plt.figure(figsize=(8,4))
    plt.plot(np.arange(L), prob(t, P), label=f"P={P:.2f}")
    plt.xlabel("posición")
    plt.ylabel("probabilidad")
    plt.legend()
    plt.tight_layout()


# —————————————————————————
# 4) Ejemplo de uso
# —————————————————————————
if __name__ == "__main__":
    # graficar después de 20 pasos con mezcla P=0.5
    plot_prob(20, 1)
    plt.show()