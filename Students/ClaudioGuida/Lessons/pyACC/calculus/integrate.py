import numpy as np
class Integrate:
    def trapezoid(func, low, up,delta=0.5): #function, lower and upper value at which compute integral, dx default=1
        x = np.arange(low, up +delta, delta) #sample grid, up+delta_x because np.arange does not include last step
        fx = func(x) #compute the function at each value of x
        return delta*((fx[0]+fx[-1])/2. + np.sum(fx[1:-1]))

    def simpson(func, low, up, n=1000):
        x = np.linspace(low,up,n+1)
        h=(low+up)/n
        fx=func(x)
        return (h/3)*(fx[0] + 4*np.sum(fx[1:n:2]) + 2*np.sum(fx[2:(n-1):2]) + fx[-1])
    
    def adaptive_quadrature(func, a, b, tol=1e-6):
        h = b - a
        s = (h/6) * (func(a) + 4*func(a+h/2) + func(b)) #simpson's rule approximation of 1/6
        def refine(a, b, h, s):
            #Refine the approximation in the interval [a, b].
            c = a + h / 2
            s1 = (h / 12) * (func(a) + 4 * func(c - h / 4) + func(c))
            s2 = (h / 12) * (func(c) + 4 * func(c + h / 4) + func(b))
            if abs(s1 + s2 - s) < tol:
                return s1 + s2 + (s1 + s2 - s) / 15
            else:
                return refine(a, c, h/2, s1) + refine(c, b, h/2, s2)
        
        return refine(a, b, h, s)