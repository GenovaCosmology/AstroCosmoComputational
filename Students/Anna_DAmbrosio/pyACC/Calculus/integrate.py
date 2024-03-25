import scipy
import scipy.integrate as int
import numpy as np

''', args=(), full_output=0, epsabs=1.49e-08, epsrel=1.49e-08, limit=50, points=None, weight=None, wvar=None, wopts=None, maxp1=50, limlst=50, complex_func=False'''

def intgr(f, x1, x2, args=(), full_output=0, epsabs=1.49e-08, epsrel=1.49e-08, limit=50, points=None, weight=None, wvar=None, wopts=None, maxp1=50, limlst=50, complex_func=False):
    
    bool = callable(f)

    if bool==True:
        #print("The integration is via scipy.integrate.quad")
        return int.quad(f, x1, x2, args=(), full_output=0, epsabs=1.49e-08, epsrel=1.49e-08, limit=50, points=None, weight=None, wvar=None, wopts=None, maxp1=50, limlst=50, complex_func=False)
    else:
        #print("The integration is via the trapezoid method")
        x = np.linspace(x1, x2, f.size)
        return np.sum((f[1:]+f[0:-1])/2 * (x[1:]-x[0:-1]))
    
    