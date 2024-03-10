import numpy as np
from ..Calculus import integral,trapezoid
import scipy
import scipy.integrate as integrate

Ms = 1.9e30 #kg

# rho unit of measurement Ms/kpc^3

def rho_DM(r,rho_0=1.4e7,rs=16.1):
    return rho_0/((r/rs)*(1+r/rs)**2)