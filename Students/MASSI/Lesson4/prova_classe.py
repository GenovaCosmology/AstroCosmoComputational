import sys
import matplotlib.pyplot as plt
import numpy as np
import math

sys.path.append("/Users/massimilianochella/AstroCosmoComputational/Students/MASSI")

# Import the Eval_Integ_Deriv class
from pyACC.Calculus.eval_integ_deriv import Eval_Integ_Deriv

# Define a new function
def another_function(x):
    return math.sin(x)

# Crea un'istanza di Eval_Integ_Deriv con la nuova funzione
valutatore = Eval_Integ_Deriv(another_function)


# Valuta l'integrale della nuova funzione da 1 a 5 con passo 0.01
valore_integrale = valutatore.integral_1D(0,math.pi,0.01)

# derivata
deriv1, x1 = valutatore.derivative_1D(1, 30, 0.01, 2)
deriv2, x2 = valutatore.derivative_1D(1, 30, 0.01, 4)
 

# Valuta la funzione
eval, x4 = valutatore.evaluate_1D(1, 30, 0.01)

print('integrale:',valore_integrale)

plt.plot(x4, eval, marker='o', linestyle='-')
plt.title('f')
plt.grid(True)
plt.show()

plt.plot(x1, deriv1, marker='o', linestyle='-')
plt.title('derivata accuratezza 2')
plt.grid(True)
plt.show()

plt.plot(x2, deriv2, marker='o', linestyle='-')
plt.title('derivata accuratezza 4')
plt.grid(True)
plt.show()




# PER FUNZIONI A PIU' VARIABILI
def sample_function(X):
    return X[0]**2 + X[1]**2 + X[2]**4
x = [1,1,1]

central_diff = Eval_Integ_Deriv(sample_function)

# Funzione valutata in x
print("Funzione valutata in ", x , "= f(", x,") = ", central_diff.evaluate_ND(x))

# Compute the derivative with respect to the first variable at the given point
print("Gradiente in ", x, " = f(",x,") = ", central_diff.derivative_ND(x, 0.001))

# Integrale della funzione nell'iperquadrato di lato l (dimensione???)