import numpy as np
import scipy
from scipy.integrate import quad
from sympy import *
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline

class Interpolation:

    """
    This class make interpolations
    """


    def __init__(self,func,x_i,x_f,steps=100,samples=1000):
        
        """
        self makes the class see all the function defined in the class
        """
        x_err = np.linspace(x_i,x_f,samples) #function range
        self.x_err = x_err
        x = np.linspace(x_i,x_f,steps)    #interpolation range
        self.x = x
        self.func = func         #theoretical function that I have interpolate
        y = [self.func(z) for z in self.x]     #theoretical function in the interpolation range
        self.y = y

   
    def lin_int(self):      
        return interp1d(self.x,self.y)   #this is the function
    #then you have to define a new function defined in the function range temp=interp1d()(x_err)

    def deltad_lin(self):
        temp_t = [self.func(z) for z in self.x_err]
        temp_i = interp1d(self.x,self.y)(self.x_err)
        return np.abs(temp_i - temp_t)


    def errlin_th(self,deltax_i,deltax_f):   #array
        temp = lambda z : self.func(z)
        #temp1 = lambda z : diff(temp,z,2) #absolute value of the second derivative of Dc
        #temp1 = lambda z : diff(temp,z,2)
        temp1 = diff(temp,z,2)
        x_temp = np.arange(deltax_i,deltax_f,0.001)
        temp2 = lambda z : temp1(z)
        temp3 = [temp2(z) for z in x_temp]
        max = np.max(np.abs(temp3))
        return max*(deltax_f - deltax_i)**2/8


    def spline(self):
        return CubicSpline(self.x,self.y)


    def deltad_spl(self):
        temp_t = [self.func(z) for z in self.x_err]
        temp_i = CubicSpline(self.x,self.y)(self.x_err)
        return np.abs(temp_i - temp_t)
