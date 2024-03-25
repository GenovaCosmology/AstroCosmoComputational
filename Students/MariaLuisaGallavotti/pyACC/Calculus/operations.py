import math as m
import numpy as np
from scipy.optimize import approx_fprime #the approx_prime function of SciPy is designed to approximate the gradient (or derivative) of a scalar-valued function of one or more variables using finite differences
from scipy.integrate import nquad

class Operations: #class definition
    def __init__(self,function): #attributes
        self.function=function

    def evaluate(self,point): #method: evaluates the function (self) in "point" (a parameter of the method evaluate, it has to be a list)
        return self.function(point)
    '''
    def integrate(self,i,f,d): #method: compute the integral of the function (self) from "i" to "f", with "d" divisions
        if len(i)!=len(f) or len(i)!=len(d):
            raise ValueError("Dimensions of 'i', 'f' and 'd' must mach.")
        
        dim=len(i)
        for j in range(dim):
            i_j=i[j]
            f_j=f[j]
            d_j=d[j]
            step=(f_j-i_j)/d_j

            while i_j<f_j:
                 self.function=integral(self.function,i_j,min(i_j+step,f_j),0.0001)
                 i_j+=step
    '''
    '''
            for k in range(dim_d):
                self.function=integral(self.function,dim_i,dim_f+step,0.0001)
                dim_i+=step
    '''
    '''
        return self.function
    '''
    def integrate(self,limits): #limits has to be a list of tuples, for example in 1D [(0,1)]
         integral,_=nquad(self.function,limits)
         return integral
    
    def differentiate(self,point,eps=1e-8): #method: compute the derivative (gradient or Jacobian) of the function (self) in "point"
        n=len(point) #number of variables
        if isinstance(self.function(point),(int,float)): #if the function in "point" is a scalar
            return approx_fprime(point,self.function,epsilon=eps) #approximates the gradient (which represents the approximation of the derivative) of the scalar-valued function "self.function" in "point", while eps is the step size used in the finite difference approximation
        else: #if the function in "point" is a vector (with N components)
            N=len(self.function(point))
            jacobian=[]
            for i in range(N):
                jacobian_row=approx_fprime(point,lambda x: self.function(x)[i],epsilon=eps)
                jacobian.append(jacobian_row)
            return jacobian