import numpy as np 

#Prueba de Dirichlet
alpha=[10,1,1,1,1,1]
muestra=np.random.dirichlet(alpha)
print(muestra)
print(muestra.sum())

def Switch(v):
    # 1) Muestreamos L-1 pesos en el simplejo (Dirichlet(1,...,1))
    alpha = np.ones(len(v))
    pesos = np.random.dirichlet(alpha)  # vector de tamaño L-1, suma 1
    
    # 2) Construimos la suma ponderada de permutaciones
    x = np.zeros_like(v, dtype=float)
    for i in enumerate(pesos):
        vp = permutar_vector(v, i, i+1)
        x += pesos[i] * vp
    
    # Por construcción, x.sum() == v.sum() == 1
    return x