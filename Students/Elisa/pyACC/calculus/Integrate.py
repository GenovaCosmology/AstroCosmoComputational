import numpy as np
import scipy
import scipy.integrate as integrate


def trapezoid(integrand,low,up,delta_x=0.01):
    x = np.arange(low,up,delta_x)
    fx = integrand(x)
    return np.sum( (fx[1:] +fx[0:-1])/2 * (x[1:]-x[0:-1]))
    #return np.trapz       wrapper, modifica il metodo trapz già esistente


def integral(integrand,low,up,delta_x=0.01):
    x = np.arange(low,up,delta_x)
    fx = integrand(x)
    return integrate.quad(lambda x : integrand(x),low,up)[0]
    #return np.trapz       wrapper, modifica il metodo trapz già esistente
