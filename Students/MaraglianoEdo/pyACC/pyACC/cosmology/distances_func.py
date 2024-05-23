from ..calculus.integrate import quad
import numpy as np
from scipy.constants import c as speed_of_light

speed_of_light = speed_of_light/1000 ### speed of light from m/s to km/s

def LambdaCDMCosmology():
    """
    Returns a dictionary with the parameters of the LambdaCDM cosmology.

    Returns:
        dict: A dictionary containing the parameters of the LambdaCDM cosmology.
    """
    
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
    """
    Calculate the Hubble function as a function of redshift.
    
    Parameters:
        z (float): Redshift.
        H0 (float): Hubble constant (km/s/Mpc).
        Om_m (float): Matter density parameter.
        Om_lambda (float): Dark energy density parameter.
        Om_rad (float): Radiation density parameter.
        Om_nu (float): Neutrino density parameter.
        Om_k (float): Curvature density parameter.
        w0 (float): Dark energy equation of state parameter (constant).
        wa (float): Dark energy equation of state parameter (time-dependent).
        
    Returns:
        float: The value of the Hubble function at redshift z.
            
    """
    return H0*np.sqrt(Om_m*(1+z)**3+Om_rad*(1+z)**4+Om_k*(1+z)**2+Om_lambda*(1+z)**(3*(1+w0)))


def comoving_distance(z, H0=67, Om_m=0.319, Om_lambda=0.681, Om_rad=0, Om_nu=0, Om_k=0, w0=-1.0, wa=0.0):
    """
    Calculate the comoving distance as a function of redshift.
    
    Parameters:
        z (float): Redshift.
        H0 (float): Hubble constant (km/s/Mpc).
        Om_m (float): Matter density parameter.
        Om_lambda (float): Dark energy density parameter.
        Om_rad (float): Radiation density parameter.
        Om_nu (float): Neutrino density parameter.
        Om_k (float): Curvature density parameter.
        w0 (float): Dark energy equation of state parameter (constant).
        wa (float): Dark energy equation of state parameter (time-dependent).
        
    Returns:
        float: The value of the comoving distance at redshift z.
        
    """

    integrand = lambda z: speed_of_light/hubble_function(z, H0, Om_m, Om_lambda, Om_rad, Om_nu, Om_k , w0, wa)

    if(isinstance(z, (int, float))):
        integral = quad(integrand, 0.0, z)[0]
    else:
        integral = np.zeros(z.size)
        for i,redshift in enumerate(z):
            integral[i] = quad(integrand, 0.0, redshift)[0]
    return integral


def angular_diameter_distance(z, H0=67, Om_m=0.319, Om_lambda=0.681, Om_rad=0, Om_nu=0, Om_k=0, w0=-1.0, wa=0.0):
    """
    Calculate the angular diameter distance as a function of redshift.
    
    Parameters:
    
        z (float): Redshift.
        H0 (float): Hubble constant (km/s/Mpc).
        Om_m (float): Matter density parameter.
        Om_lambda (float): Dark energy density parameter.
        Om_rad (float): Radiation density parameter.
        Om_nu (float): Neutrino density parameter.
        Om_k (float): Curvature density parameter.
        w0 (float): Dark energy equation of state parameter (constant).
        wa (float): Dark energy equation of state parameter (time-dependent).
        
    Returns:
        float: The value of the angular diameter distance at redshift z.
        
    """
    return comoving_distance(z, H0, Om_m, Om_lambda, Om_rad, Om_nu, Om_k, w0, wa)/(1+z)

    
def luminosity_distance(z, H0=67, Om_m=0.319, Om_lambda=0.681, Om_rad=0, Om_nu=0, Om_k=0, w0=-1.0, wa=0.0):
    """
    Calculate the luminosity distance as a function of redshift.

    Parameters:
        z (float): Redshift.
        H0 (float): Hubble constant (km/s/Mpc).
        Om_m (float): Matter density parameter.
        Om_lambda (float): Dark energy density parameter.
        Om_rad (float): Radiation density parameter.
        Om_nu (float): Neutrino density parameter.
        Om_k (float): Curvature density parameter.
        w0 (float): Dark energy equation of state parameter (constant).
        wa (float): Dark energy equation of state parameter (time-dependent).

    Returns:
        float: The value of the luminosity distance at redshift z.

    """
    return comoving_distance(z, H0, Om_m, Om_lambda, Om_rad, Om_nu, Om_k, w0, wa)*(1+z)
        

def distance_modulus_from_redshift(z, H0=67, Om_m=0.319, Om_lambda=0.681, Om_rad=0, Om_nu=0, Om_k=0, w0=-1.0, wa=0.0):

    """
    Calculate the distance modulus as a function of redshift.
    
    Parameters:
        z (float): Redshift.
        H0 (float): Hubble constant (km/s/Mpc).
        Om_m (float): Matter density parameter.
        Om_lambda (float): Dark energy density parameter.
        Om_rad (float): Radiation density parameter.
        Om_nu (float): Neutrino density parameter.
        Om_k (float): Curvature density parameter.
        w0 (float): Dark energy equation of state parameter (constant).
        wa (float): Dark energy equation of state parameter (time-dependent).
    
    Returns:
        float: The value of the distance modulus at redshift z.
    """
    # Calculate luminosity distance (in Mpc)
    dL = luminosity_distance(z, H0, Om_m, Om_lambda, Om_rad, Om_nu, Om_k, w0, wa)

    # Calculate distance modulus
    distance_modulus = 5.0 * (np.log10(dL) - 1.0) 

    return distance_modulus
    

def distance_modulus_from_luminosity_distance(luminosity_distance):
    """
    Calculate the distance modulus from the luminosity distance.

    Parameters:
        luminosity_distance (float): Luminosity distance (Mpc).

    Returns:
        float: The value of the distance modulus.
    """

    # Calculate distance modulus
    distance_modulus = 5.0 * (np.log10(luminosity_distance) - 1.0) 

    return distance_modulus

def hubble_distance(redshift):
    """
    Calculate the Hubble distance as a function of redshift.
    
    Parameters:
        redshift (float): Redshift.
        
    Returns:
        float: The value of the Hubble distance at redshift z.
    """
    return speed_of_light/hubble_function(redshift)


