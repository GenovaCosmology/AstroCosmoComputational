from ..calculus import integrate
import numpy as np

c=299792 #c in  km/s

#comoving distance
def comoving_distance(z,H_0 = 67.5 ,omega_m = 0.315, omega_r=1e-4, omega_k= 0, w=-1):
    inv_Hz= lambda z: c/(H_0*np.sqrt(omega_m*(1+z)**3 + omega_r*(1+z)**4 + omega_k*(1+z)**2 + (1-omega_m - omega_r)**(3*(1+w))))
    return integrate.adaptive_quadrature(inv_Hz, 0, z)

#angular diameter distance: DM/(1+z)
def angular_distance(z,H_0 = 67.5 ,omega_m = 0.315, omega_r=1e-4, omega_k= 0, w=-1):
    com_dist=comoving_distance(z,H_0 ,omega_m , omega_r, omega_k,w)
    return com_dist/(1+z)

#luminosity distance: DL*(1+z)
def luminosity_distance(z,H_0 = 67.5 ,omega_m = 0.315, omega_r=1e-4, omega_k= 0, w=-1):
    com_dist=comoving_distance(z,H_0 ,omega_m , omega_r, omega_k,w)
    return (1+z)*com_dist

#distance modulus
def distance_modulus(z,H_0 = 67.5 ,omega_m = 0.315, omega_r=1e-4, omega_k= 0, w=-1):
    com_dist=comoving_distance(z,H_0 ,omega_m , omega_r, omega_k,w)
    return 5*np.log10(((1+z)*com_dist)/(10**(-5)))


    