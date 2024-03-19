import numpy as np
from scipy.integrate import quad

class Integrate:
    """
    This class provides a set of methods to perform numerical integration
    """

    def __init__(self, integrand, **kwargs):
        """
        Initialize the class with the integrand function

        Parameters
        ----------
        integrand : function
            The integrand function
        kwargs : dict
            Additional arguments to be passed to the integrand function
        """
        self.integrand = lambda x : integrand(x, **kwargs)

    def trapezoidal(self, a, b, N):
        """
        Perform the integration of the integrand function between a and b
        using the trapezoidal rule with N intervals
        """
        x = np.linspace(a, b, N+1)
        y = self.integrand(x)
        h = (b - a)/N
        I = h*(0.5*y[0] + 0.5*y[-1] + np.sum(y[1:-1]))
        return I
    
    def quadrature(self, a, b, **kwargs):
        """
        Perform the integration of the integrand function between a and b
        using the quadrature method
        """
        I, _ = quad(self.integrand, a, b, **kwargs)
        return I

    def __call__(self, method, a, b, **kwargs):
        """
        Perform the integration of the integrand function between a and b

        Parameters
        ----------
        method : str
            The method to be used for the integration
        a : float
            The lower limit of the integration
        b : float
            The upper limit of the integration
        kwargs : dict
            Additional arguments to be passed to the integration method
        """

        if method=='trapezoidal':
            return self.trapezoidal(a, b, **kwargs)
        elif method=='quadrature':
            return self.quadrature(a, b, **kwargs)
        else:
            raise ValueError('Unknown integration method')
        
