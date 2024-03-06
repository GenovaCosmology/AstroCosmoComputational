import numpy as np
import scipy as sp

def trapz(f, min, max, step):
    x=np.arange(min, max, step)
    integral = np.trapz(f(x), x)
    return integral

def simpson(f,x):
    integral = sp.integrate.simpson(f(x),x)
    return integral