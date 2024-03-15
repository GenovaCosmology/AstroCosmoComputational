#I wanna to create a class that takes a function as an argument and returns the value of the function in a point or in an array
#and compute the integral of the function in a given interval and the derivative of the function in a point or a grid

from .integrate import *
from scipy.misc import derivative

class MyFunc:
    def __init__(self, func):  # Constructor method
        self.func = func  # Instance attribute

    def value(self, x):
        return self.func(x)

    def integral(self, a, b):
        return integrate.adaptive_quadrature(self.func, a, b)

    def derivative(self, x, h=1e-5, n_der=1):
        return derivative(self.func, x, dx=h, n=n_der)

