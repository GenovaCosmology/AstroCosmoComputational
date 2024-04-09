from .integrate import intgr
import scipy as sp
import numpy as np

class Function:

# 1-dimensional function

    def __init__(self,f):
        bol = callable(f)
        if bol==True:
            self.fun = f
            self.Nvar = f.__code__.co_argcount
        else:
            raise TypeError("The object is not a function.")


    def Integrate(self, x1, x2, args=(), full_output=0, epsabs=1.49e-08, epsrel=1.49e-08, limit=50, points=None, weight=None, wvar=None, wopts=None, maxp1=50, limlst=50, complex_func=False):
        if self.Nvar == 1:
            return intgr(self.fun, x1, x2, args, full_output, epsabs, epsrel, limit, points, weight, wvar, wopts, maxp1, limlst, complex_func)
        else:
            # x1 and x2 are the array containg respectively the lower and upper extremes fot the integration
            list = []
            for i, x1_i in enumerate(x1):
                list.append([x1_i, x2[i]])
            return sp.integrate.nquad(self.fun, list)[0]


    def Derivative(self, x0, dx, n=1, args=(), order=3, axis=-1, prepend="no value", append="no value"):
        
        bol = callable(self.fun)

        if bol==True:
            if type(x0)==float or type(x0)==int:
                return sp.misc.derivative(self.fun, x0, dx, n, args, order)
            else:
                deriv = np.zeros(x0.size)
                for i,xi in enumerate(x0):
                    deriv[i] = sp.misc.derivative(self.fun, xi, dx, n, args, order)
                return deriv
        else:    
            return np.diff(self.fun, n, axis, prepend, append)


    def Value(self, x0):
        if self.Nvar == 1:
            return self.fun(x0)
        
        else:
            # x0 must be a collection of arrays of the same size each containing the value of each variable
            return self.fun(*x0)