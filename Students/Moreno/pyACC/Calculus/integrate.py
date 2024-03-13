import numpy as np

def trapezoid(integrand, low ,up, delta_x):

    x = np.arange(low, up, delta_x)
    fx = integrand(x)

    return (np.sum( ( fx[1:]+fx[0:-1])/2 * (x[1:]-x[0:-1]) ))

"""
Parametri: (da rimettere)

"""