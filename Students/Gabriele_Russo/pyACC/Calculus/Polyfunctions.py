import sys
sys.path.append("../")
from Calculus import trapezoid
from Calculus import simpsons_rule
from scipy import interpolate

##Only for test programm
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",})

class Polyfunctions:
    ##************************************##
    ##Inizialization method (constructure)
    ##************************************##
    def __init__(self, function):
        self.function = function #self is telling the class to instantiate an object wich is the function I pass as parameter
    
    ##********************##
    ##Methods of the class##
    ##********************##
        
    #Evaluating function in one point
    def evaluate(self, argument):
        return self.function(argument)
    
    #Integrating function 
    def integrate_function_trapezoid(self, x_min, x_max, delta_x):
        return trapezoid(self.function, x_min, x_max, delta_x)
    
    def integrate_function_simpson_rule(self, x_min, x_max, n): #n is the number of intervalls
        return simpsons_rule(self.function, x_min, x_max, n)
    
    #Derive function at first order

    #Foward derivative method
    def forward_difference_derivative(self, x, h):
        return (self.function(x + h) - self.function(x)) / h
    
    #Central derivative method
    def central_difference_derivative(self, x, h):
        return (self.function(x + h) - self.function(x - h)) / (2 * h)
    
    #Derive function at second order

    #Central derivative method
    def central_difference_2_derivative(self, x, h):
        return (self.function(x + h) - 2 * self.function(x) + self.function(x - h)) / (h**2)
    
    #EXTENSION IN n DIMENSIONS
    def general_central_difference_derivative(self, x, h):
        """
        Compute the derivative of the function at the point x using the central difference method.

        Parameters:
        - x: array-like, the point at which to compute the derivative
        - h: float, the step size for numerical differentiation

        Returns:
        - derivative: array-like, the numerical derivative of the function at the point x
        """
        n = len(x)
        derivative = np.zeros(n)
        
        for i in range(n):
            x_plus_h = np.array(x)
            x_plus_h[i] += h
            x_minus_h = np.array(x)
            x_minus_h[i] -= h
            derivative[i] = (self.function(x_plus_h) - self.function(x_minus_h)) / (2 * h)
        
        return derivative
    
    ##**************************************##
    ##INTERPOLATION (EXERCISE 2 OF LESSON 3)##
    ##**************************************##
    #How to estimate interpolation error VS theoretical count
    def interpol_error(self, new_points, theory_points):
        
        Delta = []
        
        for i in range(len(new_points)):
            delta = new_points[i] - theory_points[i]
            Delta.append(-delta)

        return Delta
    
    
##**************************************##
##INTERPOLATION (EXERCISE 2 OF LESSON 3)##
##**************************************##
#Personal method to interpolate defined as an indipendent function
def my_interpol1d(x, y):
     return interpolate.interp1d(x, y, kind='linear', bounds_error='false', fill_value=np.nan)





        