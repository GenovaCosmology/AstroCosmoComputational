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

def hubble_function(z, H0=67, Om_m=0.319, Om_lambda=0.681, Om_rad=0, Om_nu=0, Om_k=0, w0=-1.0):
    return H0*np.sqrt(Om_m*(1+z)**3+Om_rad*(1+z)**4+Om_k*(1+z)**2+Om_lambda*(1+z)**(3*(1+w0)))


class CosmologicalDistances:
    """
    A class to compute cosmological distances using different distance measures.

    Attributes:
        hubble_function (function): A function representing the Hubble parameter as a function of redshift.
    """
    def __init__(self, hubble_function, **kwargs):
        """
        Initializes the CosmologicalDistances class.

        Args:
            hubble_function (function): A function representing the Hubble parameter as a function of redshift.
            **kwargs: Additional keyword arguments to be passed to the hubble_function.
        """
        self.hubble_function = lambda z: hubble_function(z, **kwargs)
        
                
    def comoving_distance(self, z):
        """
        Computes the comoving distance at given redshift(s).

        Args:
            z (float or list of floats): Redshift(s) at which to compute the comoving distance.
            step (float, optional): Step size for numerical integration. Defaults to 0.01.

        Returns:
            float or list of floats: Comoving distance(s) corresponding to the input redshift(s).
        """

        integrand = lambda z: speed_of_light/self.hubble_function(z)

        integral = np.zeros(z.size)
        for i,redshift in enumerate(z):
            integral[i] = quad(integrand, 0.0, redshift)[0]
        return integral

    def angular_diameter_distance(self,z):
        """
        Computes the angular diameter distance at given redshift(s).

        Args:
            z (float or list of floats): Redshift(s) at which to compute the angular diameter distance.
            step (float, optional): Step size for numerical integration. Defaults to 0.01.

        Returns:
            float or list of floats: Angular diameter distance(s) corresponding to the input redshift(s).
        """
        return self.comoving_distance(z)/(1+z)

    
    def luminosity_distance(self,z):
        """
        Computes the luminosity distance at given redshift(s).

        Args:
            z (float or list of floats): Redshift(s) at which to compute the luminosity distance.
            step (float, optional): Step size for numerical integration. Defaults to 0.01.

        Returns:
            float or list of floats: Luminosity distance(s) corresponding to the input redshift(s).
        """
        return self.comoving_distance(z)*(1+z)
        
    def distance_modulus_from_redshift(self, redshift):
        """
        Calculate the distance modulus given the redshift and absolute magnitude.

        Args:
            redshift (float): The redshift of the astronomical object.
            absolute_magnitude (float): The absolute magnitude of the astronomical object.

        Returns:
            float: The distance modulus.
        """

        # Calculate luminosity distance (in Mpc)
        luminosity_distance = self.luminosity_distance(redshift)

        # Calculate distance modulus
        distance_modulus = 5.0 * (np.log10(luminosity_distance) - 1.0) 

        return distance_modulus
    
    def distance_modulus_from_luminosity_distance(self, luminosity_distance):
        """
        Calculate the distance modulus given the redshift and absolute magnitude.

        Args:
            redshift (float): The redshift of the astronomical object.
            absolute_magnitude (float): The absolute magnitude of the astronomical object.

        Returns:
            float: The distance modulus.
        """

        # Calculate distance modulus
        distance_modulus = 5.0 * (np.log10(luminosity_distance)-1) 

        return distance_modulus

    def hubble_distance(self, redshift):
        return speed_of_light/self.hubble_function(redshift)

    def transverse_comoving_distance(self, redshift):
        return self.comoving_distance(redshift)
    
    def isotropic_volume_distance(self,redshift):
        return np.power(redshift * self.transverse_comoving_distance(redshift)**2 * self.hubble_distance(redshift), 1./3)
