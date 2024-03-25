import scipy.integrate as integrate
import numpy as np

def quad_integral(integrand, low, up):
    """
    Function to compute a minimal version of the scipy.integrate.quad method.

    Parameters
    ----------
    integrand: function
        Function to integrate.
    low: float
        Lower limit of integration
    up: float
        Upper limit for integration
    """
    return integrate.quad(integrand, low, up)[0]

#def trapezoid_integral(integrand, low, up, delta_x):
   # x = np.arange(low, up, delta_x)
   # fx= integrand(x)
   # return integrate.trapezoid(fx,low,up,)