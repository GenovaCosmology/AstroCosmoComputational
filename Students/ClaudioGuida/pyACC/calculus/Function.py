#I wanna to create a class that takes a function as an argument and returns the value of the function in a point or in an array
#and compute the integral of the function in a given interval and the derivative of the function in a point or a grid
#try to extend it to mulidimensional functions

from ..calculus import integrate
from scipy.misc import derivative
from scipy.integrate import nquad

class MyFunc:
    def __init__(self, func):  # Constructor method
        self.func = func  # Instance attribute

    def value(self, *args):
        return self.func(*args)

# it compute the integral only on the first variable
#I want to compute the integral on the second variable, 
    def integral(self, bounds, args = None):
        return nquad(self.func, bounds, args=args)
    
    def derivative(self, point, h=1e-5, n_der=1, args = None):
        return derivative(self.func, point, dx=h, n=n_der, args=args)


#I wanna to implement a method that compute partial derivative of the function
    def partial_derivative(self, point, var_index, h=1e-5, n_der=1, args=None):
        def partial_func(args=args):
            new_args = list(args)
            new_args[var_index] += h
            return (self.func(new_args) - self.func(args)) / h
        return derivative(partial_func, point, dx=h, n=n_der, args=point)