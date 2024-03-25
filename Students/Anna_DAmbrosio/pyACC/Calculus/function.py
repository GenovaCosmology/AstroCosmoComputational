from .integrate import intgr

class Function:
    def __init__(self,f):
        self.fun = f
    
    def Integrate(self, x1, x2, args=(), full_output=0, epsabs=1.49e-08, epsrel=1.49e-08, limit=50, points=None, weight=None, wvar=None, wopts=None, maxp1=50, limlst=50, complex_func=False):
        return intgr(self.fun, x1, x2, args=(), full_output=0, epsabs=1.49e-08, epsrel=1.49e-08, limit=50, points=None, weight=None, wvar=None, wopts=None, maxp1=50, limlst=50, complex_func=False)
    
    def Derivative(self, x0, dx, n=1, args=(), order=3):
        
        return