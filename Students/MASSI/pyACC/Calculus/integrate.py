import math
import numpy as np
def integrale_trapezio(funzione, min, max, dx):
    x=np.arange(min, max, dx)
    fx=[]
    tot=0
    for i in x:
        tot=tot+funzione(i)+funzione(i+dx)
    return dx*tot/2


#print (integrale_trapezio(math.sin, 0, math.pi, 0.01))