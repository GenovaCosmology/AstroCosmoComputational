from ..calculus.integrate import quad
import numpy as np
from scipy.constants import c as speed_of_light

speed_of_light = speed_of_light/1000 ### speed of light from m/s to km/s

def LambdaCDMCosmology():
    lambdaCDM_parameters = {
    'H0': 67,            # Hubble constant (km/s/Mpc)
    'Om_m': 0.319,      # Matter density parameter
    'Om_lambda': 0.681, # Dark energy density parameter
    'Om_rad': 0.0,
    'Om_nu': 0.0,
    'Om_k': 0.0,      
    'w0': -1.0,            # Dark energy equation of state parameter (constant)
    'wa': 0.0              # Dark energy equation of state parameter (time-dependent)
    }
    return lambdaCDM_parameters

def hubble_function(z, H0=67, Om_m=0.319, Om_lambda=0.681, Om_rad=0, Om_nu=0, Om_k=0, w0=-1.0, wa=0.0):
    return H0*np.sqrt(Om_m*(1+z)**3+Om_rad*(1+z)**4+Om_k*(1+z)**2+Om_lambda*(1+z)**(3*(1+w0)))


def comoving_distance(z, H0=67, Om_m=0.319, Om_lambda=0.681, Om_rad=0, Om_nu=0, Om_k=0, w0=-1.0, wa=0.0):

    integrand = lambda z: speed_of_light/hubble_function(z, H0, Om_m, Om_lambda, Om_rad, Om_nu, Om_k , w0, wa)

    if(isinstance(z, (int, float))):
        integral = quad(integrand, 0.0, z)[0]
    else:
        integral = np.zeros(z.size)
        for i,redshift in enumerate(z):
            integral[i] = quad(integrand, 0.0, redshift)[0]
    return integral


def angular_diameter_distance(z, H0=67, Om_m=0.319, Om_lambda=0.681, Om_rad=0, Om_nu=0, Om_k=0, w0=-1.0, wa=0.0):
    return comoving_distance(z, H0, Om_m, Om_lambda, Om_rad, Om_nu, Om_k, w0, wa)/(1+z)

    
def luminosity_distance(z, H0=67, Om_m=0.319, Om_lambda=0.681, Om_rad=0, Om_nu=0, Om_k=0, w0=-1.0, wa=0.0):
    return comoving_distance(z, H0, Om_m, Om_lambda, Om_rad, Om_nu, Om_k, w0, wa)*(1+z)
        

def distance_modulus_from_redshift(z, H0=67, Om_m=0.319, Om_lambda=0.681, Om_rad=0, Om_nu=0, Om_k=0, w0=-1.0, wa=0.0):

    # Calculate luminosity distance (in Mpc)
    dL = luminosity_distance(z, H0, Om_m, Om_lambda, Om_rad, Om_nu, Om_k, w0, wa)

    # Calculate distance modulus
    distance_modulus = 5.0 * (np.log10(dL) - 1.0) 

    return distance_modulus
    

def distance_modulus_from_luminosity_distance(luminosity_distance):

    # Calculate distance modulus
    distance_modulus = 5.0 * (np.log10(luminosity_distance) - 1.0) 

    return distance_modulus

def hubble_distance(redshift):
    return speed_of_light/hubble_function(redshift)


