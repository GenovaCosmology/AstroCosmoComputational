import scipy.integrate as integrate

def quadratic_integration(integrand, low, up):
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
    result = integrate.quad(integrand, low, up)

    return result

