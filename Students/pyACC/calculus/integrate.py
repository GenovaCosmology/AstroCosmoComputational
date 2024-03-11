import numpy as np
import scipy as sp

def quad(f, min, max):
    integral = sp.integrate.quad(f, min, max)
    return integral

def simpson(f,x):
    integral = sp.integrate.simpson(f(x),x)
    return integral