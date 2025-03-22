import hiperwalk as hw
import numpy as np
import time

start_time=time.time()
n=200
line=hw.Line(n)


qw=hw.Coined(line, coin= "hadamard")

v=n//2

state_i=(qw.ket((v,v+1))+1j*qw.ket((v,v-1)))/np.sqrt(2)


state_f=qw.simulate(range=(n//2,n//2+1),state=state_i)

prob = qw.probability_distribution(state_f)

hw.plot_probability_distribution(prob,plot="line") 

end_time=time.time()

tiempo=end_time-start_time

print(tiempo)