
from ..calculus.integrate import quadratic_integration
import numpy as np

def compute_flat_comoving_distance(z, H0, Omega_m):
    """
    z: redshift at which I want to evaluate the comoving distance
    H0: Hubble parameter today
    Omega_m: omega matter adimensional density
    """
    integrand = lambda x: 300000/(H0*np.sqrt(Omega_m*(1+x)**3 + 1 - Omega_m))
    return quadratic_integration(integrand, 0, z)

def comoving_distance_from_H(H, z, H0, Omega_m):
    integrand = lambda x: 300000/H(x, H0, Omega_m)
    return quadratic_integration(integrand, 0, z)[0]

