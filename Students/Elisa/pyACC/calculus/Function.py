import numpy as np
import scipy
import scipy.integrate as integrate
from sympy import *


class Function:

    """
    This class can imput different operations on an input lambda
    """


    def __init__(self,func,x0,x_min,x_max,deltax):
        
        """
        self makes the class see all the function defined in the class
        """
        
        self.func = func
        
        
        def f(self,x0):
            myfunc = lambda t : self.func(t)
            return myfunc(x0)


        def integrate(self,x_min,x_max):
            myfunc = lambda t : self.func(t)
            return integrate.quad(myfunc, x_min,x_max)


        def derivative(self,n):
            # n = derivative's degree
            myfunc = lambda t : self.func(t)
            return diff(myfunc,t,n)
        