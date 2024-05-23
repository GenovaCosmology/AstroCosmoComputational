from scipy import integrate
import matplotlib.pyplot as plt

class My1DFunction:
    """
    A class representing a one-dimensional function.

    Attributes:
        f (function): The function to be evaluated.

    Methods:
        eval(x0): Evaluates the function at a given point.
        integrate(a, b): Computes the definite integral of the function over the interval [a, b].
        diff(x0, h=0.0001, n=1, acc=1): Computes the derivative of the function at a given point.
        plot(x_array): Plots the function over a given array of x values.
    """
    def __init__(self, function):
        self.f = function

    def eval(self, x0):
        """
        Evaluates the function at a given point.
        
        Parameters:
            x0 (float): The point at which the function is to be evaluated.
            
        Returns:
            float: The value of the function at x0.
        """
        return self.f(x0)

    def integrate(self, a, b):
        """
        Computes the definite integral of the function over the interval [a, b].
        
        Parameters:
            a (float): The lower bound of the integral.
            b (float): The upper bound of the integral.
            
        Returns:
            float: The value of the integral.    
        """
        return integrate.quad(self.f, a, b)
    
    def diff(self, x0, h=0.0001, n=1, acc=1):
        """
        Computes the derivative of the function at a given point.

        Parameters:

            x0 (float): The point at which the derivative is to be computed.
            h (float): The step size for the finite difference method.
            n (int): The order of the derivative.
            acc (int): The order of accuracy of the finite difference method.


        Returns:
            float: The value of the derivative at x0.
        """
        if(n==1):
            if(acc==1):
                central_der = (self.f(x0+h)-self.f(x0-h))/(2*h)
            #if(acc==2):
            #    central_der = 1/12*self.f(x0-2*h)-2/3*self.f(x0-h)+2/3*self.f(x0+h)-1/12*self.f(x0+2*h)
            return central_der
        if(n==2):
            if(acc==1):
                second_der = (self.f(x0+h)+self.f(x0-h)-2*self.f(x0))/h**2
            #if(acc==2):
            #    second_der = -1/12*self.f(x0-2*h)+4/3*self.f(x0-h)-5/2*self.f(x0)+4/3*self.f(x0+h)-1/12*self.f(x0+2*h)
   
            return second_der

    def plot(self, x_array):
        """
        Plots the function over a given array of x values.

        Parameters:
            x_array (array): An array of x values.

        Returns:
            None.
        """
        plot=plt.plot(x_array, self.f(x_array))
