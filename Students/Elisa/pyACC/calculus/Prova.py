import numpy as np
import scipy
from scipy.integrate import quad
from sympy import *


class Prova:
    
    """
    This class can imput different operations on an input lambda
    """


    def __init__(self,func):
        
        """
        self makes the class see all the function defined in the class
        """
        self.func = func
        #lambda x,y,z : func(x,y,z)

    def point(self,x0):
            #myfunc = lambda t : self.func(t)
            return self.func(x0)

'''
    def integration(self,x_min,x_max):
            myfunc = lambda t : self.func(t)
            return quad(myfunc, x_min,x_max)[0]


        def derivation(self,n):
            # n = derivative's degree
            myfunc = lambda t : self.func(t)
            return diff(myfunc,t,n)
        
'''


'''

class Operations1d:
    
    def __init__(self,func,x):
        
        """
        self makes the class see all the function defined in the class
        """
        self.func = lambda x: func(x)

        
    def point(self,x0):
        #myfunc = lambda var : self.func(var)
        #return myfunc(x0)
        return self.func(x0)


    def integration(self,min,max):
        #myfunc = lambda var : self.func(var)
        return quad(self.func,min,max)[0]


    def derivation(self,n=1):
        # n = derivative's degree
        x = symbols('x')
        myfunc = lambda t : self.func(t)
        return diff(myfunc,var,n)
'''