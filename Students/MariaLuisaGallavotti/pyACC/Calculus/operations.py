#•Differenza "in avanti":
#errore totale = errore di troncamento + errore di approssimazione (dipende da come definisco i numeri)
#devo scegliere il time step opportuno per minimizzare l'errore

#•Differenza centrale:
#l'errore è proporzionale ad h^3
#problema: non posso ottenere la derivata in tutti i punti della griglia, perchè mi servono le condizioni al contorno

#Attenzione: se h è troppo piccolo potrei avere un errore grande


#from .integrate import integral
#'''
import sys
sys.path.append("../../")
from pyACC.calculus import integral
#'''
import math as m
import numpy as np
from scipy.optimize import approx_fprime #the approx_prime function of SciPy is designed to approximate the gradient (or derivative) of a scalar-valued function of one or more variables using finite differences

class Operations: #class definition
    def __init__(self,function): #attributes
        self.function=function

    def evaluate(self,point): #method: evaluates the function (self) in "point" (a parameter of the method evaluate, it has to be a list)
        return self.function(point)
    
    def integrate(self,i,f,d): #method: compute the integral of the function (self) from "i" to "f", with "d" divisions
        if len(i)!=len(f) or len(i)!=len(d):
            raise ValueError("Dimensions of 'i', 'f' and 'd' must mach.")
        
        dim=len(i)
        for j in range(dim):
            dim_i=i[j]
            dim_f=f[j]
            dim_d=d[j]
            step=(dim_f-dim_i)/dim_d

            for k in range(dim_d):
                self.function=integral(self.function,dim_i,dim_f+step,0.0001)
                dim_i+=step

        return self.function
    
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

'''
#x
def lin(x):
    return x
x=Operations(lin)
print("Integral of x between 0 and 2:",x.integrate(0,2,0.01))
'''

'''
#sin(pi)
sin=Operations(m.sin)
if m.isclose(sin.evaluate(m.pi),0,abs_tol=1e-15):
    print("Sin(pi)=0")
else:
    print("Sin(pi)=",sin.evaluate(m.pi))
'''

'''
#sin(pi/2)
sin=Operations(m.sin)
print(sin.evaluate(m.pi/2))
'''

'''
#to verify the "differentiate" method
def function(x):
    if len(x)==1:
        return x[0]**2
    else:
        return [x[0]**2+x[1]**2,2*x[0]*x[1]]
op=Operations(function)

#scalar
x_0=[1.0] #This creates a list containing the single element 1.0. It's done this way to ensure that x_0 is iterable, which is often necessary when dealing with functions that expect input in list format
gradient=op.differentiate(x_0)
print("The derivative at ",x_0," is ",gradient)

#vector
x_0v=[1.0,2.0]
jacobian=op.differentiate(x_0v)
print("The jacobian at {} is {}".format(x_0v,np.array(jacobian)))
'''

'''
#to verify the evaluate method in more dimensions
def function(x):
    if len(x)==1:
        return x[0]**2
    else:
        return [x[0]**2+x[1]**2,2*x[0]*x[1]]
f=Operations(function)
x_0=[1.0]
print("The value of f=x^2 at {} is {}".format(x_0,f.evaluate(x_0)))

x_0v=[1.0,2.0]
print("The value of f=[x^2+y^2,2xy] at {} is {}".format(x_0v,f.evaluate(x_0v)))
'''