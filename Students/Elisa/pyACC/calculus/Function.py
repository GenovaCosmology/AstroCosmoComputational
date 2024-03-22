import numpy as np
import scipy
from scipy.integrate import quad
from sympy import *


class Function:

    """
    This class can imput different operations on an input lambda
    """


    def __init__(self,func):
        
        """
        self makes the class see all the function defined in the class
        """
        
        self.func = func
        
        
    def f(self,x0):
        myfunc = lambda t : self.func(t)
        return myfunc(x0)


    def integration(self,x_min,x_max):
        myfunc = lambda t : self.func(t)
        return quad(myfunc, x_min,x_max)[0]


    def derivation(self,n):
        # n = derivative's degree
        myfunc = lambda t : self.func(t)
        return diff(myfunc,t,n)
        