import numpy as np
import random

estado=np.array([1,0])

def caminar(estado,direccion):
    if direccion == "derecha":
        operador_X= np.array([[0,1],[1,0]])
        return np.dot(operador_X,estado)
    else:
        operador_Z= np.array([[1,0],[0,-1]])
        return np.dot(operador_Z, estado)
    
pasos =10

for i in range(pasos):
    direccion =random.choice(["derecha","izquierda"])
    estado = caminar(estado, direccion)
    print(f"Paso{i+1}:{direccion},Estado actual{estado}")
    