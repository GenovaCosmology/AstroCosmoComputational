

import numpy as np
import matplotlib.pyplot as plt
# Import function class
from .function import Funct
import scipy as sp
from scipy import interpolate
from sympy import *
import sympy as sym


# Interpolation class
class INTERP:

    def __init__(self, func):
        if callable(func)==False:
            raise ValueError("The input must be a function (Lambda)")
        self.func = Funct(func)
        #self.func_int = func
        self.Nvar = func.__code__.co_argcount
    
    def INTERP1d(self, xi, xf, Nint, x_new, method='linear', eval=True):

        '''
        This method interpolates a function in a given range [xi,xf] with Nint points (in could be a list of 
        N points) and evaluates the interpolation in x_new (if eval=True). The method can be 'linear' or 'cubic'.

        Parameters:
        - xi: initial value of the range (int or float)
        - xf: final value of the range (int or float)
        - Nint: number of points in the range (int, list or numpy array)
        - x_new: value where the interpolation is evaluated (numpy array)
        - method: method of interpolation ('linear' or 'cubic')
        - eval: if True, the interpolation is evaluated in x_new
        
        Returns:
        - interp_return_ev_list: list of the evaluation of the interpolation in x_new (if eval=True) [0]
        - interp_return_list: list of the interpolation functions [0]
        - error_list: list of the errors in the interpolation [1]
        - th_error_list: list of the theoretical errors in the interpolation [2]
        '''

        if self.Nvar!=1:
            raise ValueError("The input function must be a function of one variable, you've inserted one with ", self.Nvar, " variables")
        if type(xi)!=int and type(xi)!=float:
            if type(xf)!=int and type(xf)!=float:
                raise ValueError("The initial and final values must be integers or floats")
        if type(Nint)!=int and type(Nint)!=list and type(Nint)!=np.ndarray:
            raise ValueError("The number of points must be an integer, a list or a numpy array (of integers)")
        if xi>=xf:
            raise ValueError("The initial value must be smaller than the final value")
        
        x_inter_list=[]
        y_inter_list=[]
        interp_return_list=[]
        interp_return_ev_list=[]
        error_list = []
        th_error_list = []
        for i,nint in enumerate(Nint):
            if type(nint)!=int or nint<=0:
                raise ValueError("The number of points must be a positive integer")
            x_inter_list.append(np.linspace(xi,xf,nint))
            y_inter_list.append(self.func.Val(x_inter_list[i]))

            # interpolations
            if method=='linear':
                interp_return_list.append(sp.interpolate.interp1d(x_inter_list[i],y_inter_list[i],kind='linear'))
            elif method=='cubic':
                interp_return_list.append(sp.interpolate.interp1d(x_inter_list[i],y_inter_list[i],kind='cubic'))
            if eval==True:
                interp_return_ev_list.append(interp_return_list[i](x_new))
            # errors
            y_new=self.func.Val(x_new)
            error_list.append(np.abs(y_new-interp_return_list[i](x_new)))
            th_error_list.append((((xf/nint)**2)/8)*np.abs(self.func.Der(x_new,ord=2)))
        if eval==True:
            return interp_return_ev_list, error_list, th_error_list
        else:
            return interp_return_list, error_list, th_error_list
        
# the implementation above is useful but IT'S not a class method!!!

def interp3D_grid(data,size,Nbins):
    '''
    This function interpolates a 3D grid of data with a given size and number of bins.

    Parameters:
    - data: set of 3D data points (numpy array) (Nx3)
    - size: size of the grid (float)
    - Nbins: number of bins (int)

    Returns:
    - pdf: 3D grid of interpolated data (numpy array) (Nbins+1xNbins+1xNbins+1)
    '''
    spacing = size/Nbins
    Npoints = data.shape[0]
    pdf = np.zeros((Nbins+1,Nbins+1,Nbins+1))
    # pdf calculation
    for i in range(Npoints):
        dataX=data[i,0]
        dataY=data[i,1]
        dataZ=data[i,2]
        idx=int(dataX/spacing)
        idy=int(dataY/spacing)
        idz=int(dataZ/spacing)
        if (dataX/spacing - int(dataX/spacing)) >= 0.5:
            idx=int(dataX/spacing)+1
        if (dataY/spacing - int(dataY/spacing)) >= 0.5:
            idy=int(dataY/spacing)+1
        if (dataZ/spacing - int(dataZ/spacing)) >= 0.5:
            idz=int(dataZ/spacing)+1
            pdf[idx,idy,idz]+=1
    return pdf/Npoints