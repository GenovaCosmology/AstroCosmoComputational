import numpy as np
from ..Calculus import integral,trapezoid
import scipy
import scipy.integrate as integrate


#number of stars
def psi(M,psi_0=1,alpha=2.35):
    return psi_0*np.power(M,-alpha)