# class that computes different operations on a given function
import scipy
import numpy as np
from scipy.misc import derivative
from .integrate import integrate_f


class Function:
    def __init__(self, func):
        self.func = func
    
    def Val(self, x):
        '''
        This function computes the value of a function at a given point

        Parameters:
        x: point where the function is computed (float or array)
        '''
        if callable(self.func)== True:
            return self.func(x)
        if type(self.func)== np.ndarray or type(self.func)== float or type(self.func)== int:
            str= type(self.func)
            print("The object is not a function, it's a ",str, ", so I simply returned it!")
            return self.func
              
    def Int(self, x_in=0, x_fin=0):
        '''
        This function integrates a function:
        It uses the integrate_f function from the integrate.py file

        Parameters:
        x_in: initial point of the integration (float)

        x_fin: final point of the integration (float)
        '''
        return integrate_f(self.func, x_in, x_fin)
    
    def Der(self, x ,h=1e-5, ord=1):
        '''
        This function computes the derivative of a function

        Parameters:
        x: point where the derivative is computed (float or array)
        h: step size (float)
        ord: order of the derivative (int)
        '''
        if type(ord)!= int:
            raise TypeError("ord must be an integer")
        if callable(self.func)== True:
            if type(x)== float or type(x)== int:
                return derivative(self.func, x, dx=h,n=ord)
            if type(x)== np.ndarray:
                der_arr=np.zeros(x.size)
                for i,xi in enumerate(x):
                    der_arr[i]=derivative(self.func, xi, dx=h,n=ord)
                return der_arr
        if type(self.func)== np.ndarray:
            der_arr1=np.diff(self.func,n=ord)
            der_arr2=np.append(der_arr1,der_arr1[der_arr1.size-1])
            return der_arr2

                
