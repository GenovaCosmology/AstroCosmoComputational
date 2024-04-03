import sys
import math
import numpy as np
sys.path.append("/Users/massimilianochella/AstroCosmoComputational/Students/MASSI")
from pyACC.Calculus.integrate import integrale_trapezio


class Eval_Integ_Deriv:

    def __init__(self, f):
        self.f = f

    '''
    FUNZIONI SCALARI DA R IN R
    
        1 DIMENSIONALI
    '''

    def evaluate_1D(self, min, max, h):
        x=np.arange(min, max, h)
        fx=[]
        for i in x:
            fx.append(self.f(i))
        return fx, x                    # RESTITUISCE DUE ARRAY, UNO PER LA F(x) E UNO PER LE X

    def integral_1D(self, min, max, h):
        return integrale_trapezio(self.f, min, max, h)

    def derivative_1D(self, min, max, h, accuratezza):    # accuratezza può essere 2 o 4
        D_f = []
        x = np.arange(min, max, h)
        x_i = []
        l = np.size(x)
        fx, x = self.evaluate_1D(min, max, h)

        if accuratezza==2:
            for i in range(0,l-3):
                D_f.append((1/(2*h))*(fx[i+2]-fx[i]))
                x_i.append(x[i])

        if accuratezza==4:
            for i in range(0,l-5):
                D_f.append((1/12)*(1/h)*(fx[i]-fx[i+4])+(1/h)*(2/3)*(fx[i+3]-fx[i+1]))
                x_i.append(x[i])

        return D_f, x_i     # RESTITUISCE DUE ARRAY, UNO PER LA DERIVATA E UNO PER LE X
    


    '''
    FUNZIONI SCALARI DA RN IN R
    
        N DIMENSIONALI
    '''


    def derivative_ND(self, x, h):
        grad = []
        n = len(x)
        X_plus = []
        X_minus= []

        for i in range(0, n):
            xi_plus = []
            xi_minus = []
            for j in range (0, n):
                if i == j:
                    xi_plus.append(x[i] + h)
                    xi_minus.append(x[i] - h)
                else:
                    xi_plus.append(x[j])
                    xi_minus.append(x[j])
                
            X_plus.append(xi_plus)
            X_minus.append(xi_minus)

            grad.append((self.f(X_plus[i])-self.f(X_minus[i]))*0.5*1/h)
        

        return grad     # VALUTA LA FUNZIONE NEL PUNTO X


    def evaluate_ND(self, x):
        return self.f(x)
    
    def integral_ND():
        
        
        
        
        return