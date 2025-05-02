from scipy import sparse
import numpy as np
import matplotlib.pyplot as plt
from numpy import random

n_q=3 # número de qubits

largo=2**n_q #largo de nuestra caminata

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

for i in range(largo):
    print (i, "se mapea a", Swap_q(i,0,2,n_q))