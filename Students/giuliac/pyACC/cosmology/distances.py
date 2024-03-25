from ..calculus.integrate import quad_integral
import numpy as np

def comoving_dist(z,Om,Or,Ow,Ok,H0):
    integrand = lambda x: 300000/(H0*np.sqrt(Om*(1+x)**3+Or*(1+x)**4+Ow*(1+x)**(3*(1+w))+Ok*(1+x)**2))
    return quad_integral(integrand,0,z)

def flat_comoving_dist(z,Om,H0):
    integrand = lambda x: 300000/(H0*np.sqrt(Om*(1+x)**3+(1-Om)))#H=H0*np.sqrt(Om*(1+z)**3+(1-Om))
    return quad_integral(integrand,0,z)