import math as m
import numpy as np
#import sympy as sp
from scipy.integrate import nquad
from scipy.interpolate import interp1d

class MyFunction:
    def __init__(self,function): #self mi dice che la classe prende delle funzioni e se stessa??
        self.function = function
        #function in generally is an array of functions(so I have a multidimensional function)

    #ora creo i metodi
        
    #gave me the value of the function in a point
    def value_f(self,val):
        return self.function(val)
    
    #gave me the value of the integral of function
    def integral1(self,limits):
        result, error = nquad(lambda *args: self.function(args), limits)
        return result
    
    #gave me the value of the integral of function
    def integral(self,xmin,xmax,n):
        w=(xmax-xmin)/n #lenght of the integration's interval
        value = (self.function(xmin) + self.function(xmax))*w/2
        for i in range(0,n-1,1):
            x = xmin + w*i
            value = value + w*self.function(x)
        return value
    
    #gave me the value of the derivative's function
    def gradient(self, x):
        h = 1e-6  # Step size for numerical differentiation
        grad = np.zeros_like(x)  # Initialize gradient vector
        for i in range(len(x)):
            x_plus_h = x.copy()
            x_plus_h[i] += h
            x_minus_h = x.copy()
            x_minus_h[i] -= h
            grad[i] = (self.function(x_plus_h) - self.function(x_minus_h)) / (2 * h)
        return grad
    
    #gave me the value of the function's derivative
    #Method: Central Difference
    def derivative(self,val,h):
        return (self.function(val + h) - self.function(val - h))/(2*h)
    
    #gave me the value of the function's second derivative
    def second_derivative(self,val,h):
        return (self.function(val + h) + self.function(val - h) - 2*self.function(val))/(h**2)

#ora faccio i metodi per l'interpolazione:
    #interpolazione lineare
    def interpolation_lin(self,x_new):
        Dc_I = self.function(x_new)
        return Dc_I
    #interpolazione cubica
    def interpolation_cub(self,x_old,y_old,x_new):
        f_interpol = interp1d(x_old,y_old,kind='cubic')
        Dc_I = f_interpol(x_new)
        return Dc_I



#Proof
'''
 #for the second derivative
def funz(x):
    return x**2
fun0 = MyFunction(funz)
print(fun0.second_derivative(7,0.003))   


#for the value of the function:
def funz2D(x):
    return x[0]+m.sin(x[1])

fun4 = MyFunction(funz2D)
val = [1,2]
value = fun4.value_f(val)
print(value)

#for the integral
limits = [(0,1), (0,2)]
print(fun4.integral1(limits))
#result, error = nquad(lambda x, y: funz2D((x, y)), limits)
#result, error = nquad(lambda x, y: funz2D((x, y)), [x_limits, y_limits])
#print(result)

def funz(x):
    return m.sin(x[0])
fun1 = MyFunction(funz)
limits1 = [(0,m.pi)]
print(fun1.integral1(limits1))
point = [m.pi]
print(fun1.gradient(point))


#for the value of the function:
fun3 = MyFunction(m.cos)
print(fun3.value_f(m.pi))


#for the integral:
fun1 = MyFunction(m.sin)
print(fun1.integral(0,m.pi,1000))

def fun_linear(x):
    return x
fun2 = MyFunction(fun_linear)
print(fun2.integral(0,1,1000))

#for the derivative:
print(fun1.derivative(m.pi,0.01))
'''

'''
by chatGPT to do a derivative:

import numpy as np

def func(x):
    return x[0]**2 + x[1]**2  # Example of a simple quadratic function of two variables

def gradient(func, x):
    h = 1e-6  # Step size for numerical differentiation
    grad = np.zeros_like(x)  # Initialize gradient vector
    for i in range(len(x)):
        x_plus_h = x.copy()
        x_plus_h[i] += h
        x_minus_h = x.copy()
        x_minus_h[i] -= h
        grad[i] = (func(x_plus_h) - func(x_minus_h)) / (2 * h)
    return grad

x = np.array([1.0, 2.0])  # Point at which to compute the gradient
grad = gradient(func, x)
print("Gradient at x =", x, "is", grad)


'''