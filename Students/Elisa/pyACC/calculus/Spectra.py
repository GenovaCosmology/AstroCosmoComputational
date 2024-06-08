import numpy as np
import scipy
from scipy.integrate import nquad
from sympy import *



class Spectra:
    
    """
    This class can imput different operations on an input lambda
    """


    def __init__(self,func):
        
        """
        self makes the class see all the function defined in the class
        """
        
        self.func = func
        self.nvar = func.__code__.co_argcount

        
    def point(self,*args):
        #myfunc = lambda t : self.func(t)
        return self.func(*args)

'''
    def integration(self,lim=iterable object):
        #myfunc = lambda t : self.func(t)
        return nquad(self.func, lim)
'''
'''
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