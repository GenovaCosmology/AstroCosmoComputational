#•N•O•T•E•S•

#ALGORITHMS TO EVALUATE INTEGRALS
#•Trapezoid rule: sum of the area of given trapezoids -> the error I make is inversely proportional to the number of trapezoids I used to approximate the function, moreover it is proportional to the second derivative of the function
#•Fixed-Point quadrature method
#•Adaptive quadrature method

#N.B.: if I want to be fast I have to decide how precise I want to be (all depends on the application)

#IMF: it describes how mass is distributed
#Dark matter halos: NFW profile
#Cosmological distances (using the Hubble function)


'''
import math
def integral(function, i, f, n): #trapezoid method
    #function: function to integrate
    #i: lower limit
    #f: upper limit
    #n: number of intervals

    #I=(f(i)+f(f))*w/2+w*somma per i(1->n-2)[f(x_i)]

    w=(f-i)/n
    #x=np.arange(i,f,w) #x has values that vary from i to f with spacing w
    #f_x=function(x) #f_x is a function of the variable x
    val=(function(i)+function(f))*w/2
    for j in range(0,n,1):
        x=i+w*j
        val=val+w*function(x)
    return val

def prova(x):
    return x
print("Integral", integral(prova,0,1,1000))
'''

'''
import numpy as np
import math as m

def integral(function,i,f,d):
    #d: step size
    x=np.arange(i,f,d,dtype=float)
    func = np.vectorize(function)(x)
    return (np.sum((func[1:]+func[0:-1])/2*(x[1:]-x[0:-1])))

def prova(x):
    return x
print("Sine", integral(m.sin,0,m.pi+0.0001,0.0001))
print("x", integral(prova,0,1.0001,0.0001))

'''


#precise way using Gaussian quadrature method (SciPy)
import numpy as np
from scipy.integrate import quad
import math as m

def integral(func,i,f,d):
    #func: the function to integrate
    #i: lower limit
    #f: upper limit
    #d: spacing

    #number of subdivisions
    n = m.ceil((f-i)/d)
    
    result,error=quad(func,i,f,limit=n)
    return result

'''
#example: sine from 0 to pi
def integrand(x):
    return np.sin(x)
i = 0
f = m.pi
d = 0.01
result = integral(integrand, i, f, d)
print("Integral using quad:", result)
'''