import numpy as np
import scipy as sp

def quad(f, min, max):
    """
    Compute the definite integral of a function using the quadrature method.

    Parameters:
        f (callable): The function to be integrated.
        min (float): The lower limit of integration.
        max (float): The upper limit of integration.

    Returns:
        tuple: A tuple containing the result of the integration and an estimate of the absolute error.

    Notes:
        This function uses scipy's quad function for numerical integration.

    Example:
        To integrate the function f(x) = x**2 from 0 to 1:
        >>> result, error = quad(lambda x: x**2, 0, 1)
    """
    integral = sp.integrate.quad(f, min, max)
    return integral

def simpson(f, min, max, num_points=100):
    """
    Compute the definite integral of a function using Simpson's rule.

    Parameters:
        f (callable): The function to be integrated.
        min (float): The lower limit of integration.
        max (float): The upper limit of integration.
        num_points (int, optional): The number of points to use for numerical integration. Default is 100.

    Returns:
        float: The value of the definite integral.

    Notes:
        This function uses scipy's simpson function for numerical integration.

    Example:
        To integrate the function f(x) = x**2 from 0 to 1:
        >>> result = simpson(lambda x: x**2, 0, 1)
    """
    x = np.linspace(min, max, num_points)
    integral = sp.integrate.simps(f(x), x)
    return integral