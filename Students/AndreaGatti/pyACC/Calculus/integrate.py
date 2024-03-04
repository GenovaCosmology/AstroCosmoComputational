import numpy as np
import math as m

'''
def trapezoid(integrand, low, up, delta_x):
    x = np.arange(low, up, delta_x)
    fx = integrand(x)

    return (np.sum( (fx[1:] + fx[0:-1])/2 * (x[1:]-x[0:-1]) ))

a = trapezoid(m.sin, 0, m.pi, 0.1 )

print(a)
'''
#new code:

def trapezi(f, a, b, n):
    w = (b-a)/n 
    val = (f(a) + f(b))*w/2
    for i in range(0,n-1,1):
        x=a + w*i
        val = val + w*f(x)
    return val

valore_int = trapezi(m.sin, 0, m.pi, 10000)
print(valore_int)
