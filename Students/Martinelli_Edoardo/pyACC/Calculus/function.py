# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-* #
#    _______  _       _   __     ___    ______  _________  _____  _________   __     ___                    #
#    |        |       |   | \     |    /            |        |    |       |   | \     |                     #
#    |        |       |   |  \    |   /             |        |    |       |   |  \    |                     #
#    |____    |       |   |   \   |  |              |        |    |       |   |   \   |                     #
#    |        |       |   |    \  |   \             |        |    |       |   |    \  |                     #
#   _|_       |_______|  _|_    \_|    \______     _|_     __|__  |_______|  _|_    \_|                     #
# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-* #
#                                                                                                           #
# AUTHOR: Edoardo Martinelli                                                                                #
#                                                                                                           #
# DESCRIPTION: This file contains the Function class, which is used to create a function object that can    #
# be evaluated, integrated and differentiated. The class is initialized with a function or a numpy array    #
# (function already evaluated). The class has 4 methods: Val, Int, Der and DerN. Val computes the value     #
# of the function in a point, Int integrates it, Der computes the derivative of the function and DerN       #
# computes the derivative if the function have N variables (it must be polinomial at present).              #
#                                                                                                           #
# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-* #

import scipy
import numpy as np
from scipy.misc import derivative
from scipy.integrate import nquad
from .integrate import integrate_f
from sympy import *
import sympy as sym

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-* #
#                                              OBJECT FUNCTION                                              #                                         
# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-* #

class Funct:
    def __init__(self, func):
        if callable(func)== False and isinstance(func,sym.Basic)==False and type(func)!= np.ndarray and type(func)!= float and type(func)!= int:
            raise TypeError("func must be a function (lambda, sympy, constant int or float) or a numpy array (function already evaluated)")
        self.func = func
        if callable(func)== True:
            self.Nvar = func.__code__.co_argcount
            self.symbols = 0
        if type(func)== np.ndarray or type(func)== float or type(func)== int:
            self.Nvar=1
            self.symbols = 0
        if isinstance(func,sym.Basic)==True:
            self.Nvar = len(list(func.atoms(Symbol)))
            self.symbols = list(func.atoms(Symbol))
        print("Function object created, you've inserted a ", type(func), " function with ",self.Nvar, " variables")

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-* #
#                                             EVALUATION METHOD                                             #
# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-* #
        
    def Val(self, x):
        '''
        This function computes the value of a Nvar variables function at a given point

        Parameters:
        x: point where the function is computed (float, array or matrix (rows as array of each variables))
        '''
        if self.Nvar==1:
            if callable(self.func)== True:
                return self.func(x)
            if type(self.func)== np.ndarray or type(self.func)== float or type(self.func)== int:
                str= type(self.func)
                print("The object is not a function, it's a ",str, ", so I simply returned it!")
                return self.func
            if isinstance(self.func,sym.Basic)==True:
                func_lambda= sym.lambdify(self.symbols,self.func)
                return func_lambda(x)
        if self.Nvar>1:
            if callable(self.func)== True:
                if np.shape(x)[0]!= self.Nvar:
                   raise TypeError("Input x must be a matrix with, in each row, the values of the variables with which you want the function to be evaluated")
                else:
                    # it returns a n=(# of values of each variables) array with the values of the function evaluated at the points in x
                    return self.func(*x)
            if type(self.func)== np.ndarray or type(self.func)== float or type(self.func)== int:
                print("The object is not a function, it's a numpy array (I don't know if it's a function avaluated) or an int or a float, so I simply returned it!")
                return self.func
            if isinstance(self.func,sym.Basic)==True:
                func_lambda= sym.lambdify(self.symbols,self.func)
                return func_lambda(*x)
            
# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-* #
#                                            INTEGRATION METHOD                                             #                                         
# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-* #  
                 
    def Int(self, x_in=0, x_fin=0):
        '''
        This function integrates a function of N variables:
        It uses the integrate_f function from the integrate.py file and other implementations

        Parameters:
        x_in: initial point of the integration (float or array Nvar size)

        x_fin: final point of the integration (float or array Nvar size)
        '''
        if self.Nvar==1:
            if type(x_in)!= float or type(x_fin)!= float:
                if type(x_in)!= int or type(x_fin)!= int:
                    raise TypeError("x_in and x_fin must be floats or ints!")
                if callable(self.func)== True:
                    return integrate_f(self.func, x_in, x_fin)
                if isinstance(self.func,sym.Basic)==True:
                    # it will return a sympy object, it won't be evaluated
                    integral= sym.lambdify(self.symbols,sym.integrate(self.func, self.symbols))
                    return integral(x_fin)-integral(x_in)
        if self.Nvar>1:
            if x_fin.size!=x_in.size or x_in.size!=self.Nvar:
                raise TypeError("x_in and x_fin must have same size = Nvar!")
            if isinstance(self.func,sym.Basic)==True:
                raise TypeError("For sympy function the Int method works only for 1 variable functions!")
            else:
                ext=[]
                for i,ent in enumerate(x_in):
                    ext.append([ent,x_fin[i]])
                return nquad(self.func,ext)[0]
            
# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-* #
#                                             DERIVATIVE METHOD                                             #                                  
# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-* #   
            
    def Der(self, x ,h=1e-5, ord=1,analytEval=False):
        '''
        This function computes the derivative of a function with 1 variable

        Parameters:
        x: point where the derivative is computed (float or array)
        h: step size (float)
        ord: order of the derivative (int)
        '''
        if self.Nvar>1:
            raise TypeError("The function has more than 1 variable, use the DerN method!")
        if type(ord)!= int or ord<0:
            raise TypeError("ord must be a positive integer")
        
        if isinstance(self.func,sym.Basic)==True:
            X= self.symbols[0]
            DER=sym.diff(self.func,X,ord)
            print('The derived function is: ',DER)
            if analytEval==True:
                func_der_lam= sym.lambdify(self.symbols,DER)
                return func_der_lam(x)
            else:
                return DER
        
        if callable(self.func)== True:
            if type(x)== float or type(x)== int:
                return derivative(self.func, x, dx=h,n=ord)
            if type(x)== np.ndarray:
                der_arr=np.zeros(x.size)
                for i,xi in enumerate(x):
                    der_arr[i]=derivative(self.func, xi.item(), dx=h,n=ord)
                return der_arr
        if type(self.func)== np.ndarray:
            der_arr1=np.diff(self.func,n=ord)
            der_arr2=np.append(der_arr1,der_arr1[der_arr1.size-1])
            return der_arr2
        
# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-* # 
#                              DERIVATIVE OF A FUNCTION WITH N VARIABLES METHOD                             #                     
# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-* #  
            
    def DerN(self,xi,ni,eval=True):
        '''
        This function compute the derivative of a N variable function
        WARNING: the function must be a polinomial function (At present!)

        xi: array or matrix where you want the derivative to be computed (array or matrix (rows as array of each variables))
        ni: order of derivation for each variable (array Nvar size)
        eval: bool (optional: True); if you want the function to return a lambda function derived set eval=False
        '''
        if isinstance(self.func,sym.Basic)==False:
            raise TypeError("The function must be a sympy function!")
        if np.array(ni).size!=self.Nvar:
            raise TypeError('ni must be an array with Nvar size!')
        print('------------------------------------------------------------------')
        print('I read the variables of the function are: ',self.symbols, ' ; IN THIS ORDER!')
        print('------------------------------------------------------------------')
        print('You have inserted the function: ',self.func)
        func_der=self.func
        for i,Ni in enumerate(ni):
            print('------------------------------------------------------------------')
            print('You want it to be derived with respect to the variable ',self.symbols[i],' , ',Ni,' times.')
            print('You want then the derivative to be computed at the point ', self.symbols[i],'=',xi[i])
            func_der=func_der.diff(self.symbols[i],Ni)
            print('After this derivation the function is: ',func_der)
            print('------------------------------------------------------------------')
        func_der_lam= sym.lambdify(self.symbols,func_der)
        if eval==True:
            return func_der_lam(*xi)
        else:
            return func_der