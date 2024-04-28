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
    # I want to compute the integral on the second variable
    def integral(self, bounds, n=1, args=None):
        if n != 1:
            return nquad(self.func, bounds, args=args)[0]
        return integrate.adaptive_quadrature(self.func, bounds[0][0], bounds[0][1])

    def derivative(self, point, h=1e-5, n_der=1, *args):
        return derivative(self.func, point, dx=h, n=n_der, args=args)


#I wanna to implement a method that compute partial derivative of the function
    def partial_derivative(self, point, var_index, h=1e-5, n_der=1):
        def partial_func(x, *args):
            new_args = list(args)
            new_args[var_index] = x
            return self.func(*new_args)
        #I want to evaluate the partial derivative of the function in a point, with respect to the variable var_index, i have to use value method
        return derivative(partial_func, point[var_index], dx=h, n=n_der, args=point)
    

    def gradient(self, point, h=1e-5, n_der=1):
        return [self.partial_derivative(point, i, h, n_der) for i in range(len(point))]
    

    def partial_derivative_at_point(self, variable, point, h=1e-6):
        if not callable(self.func):
            raise ValueError("The function is not callable.")
        
        if variable not in self.func.__code__.co_varnames:
            raise ValueError(f"'{variable}' is not a variable in the function.")

        index = self.func.__code__.co_varnames.index(variable)

        def perturbed_point_plus(point):
            perturbed_point = list(point)
            perturbed_point[index] += h
            return tuple(perturbed_point)

        def perturbed_point_minus(point):
            perturbed_point = list(point)
            perturbed_point[index] -= h
            return tuple(perturbed_point)

        partial_derivative = (self.value(*perturbed_point_plus(point)) - self.value(*perturbed_point_minus(point))) / (2 * h)

        return partial_derivative