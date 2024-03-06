from ..calculus.integrate import trapz, simpson
import numpy as np
from scipy.constants import c as speed_of_light

speed_of_light = speed_of_light/1000 ### speed of light from m/s to km/s

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
        
                
    def comoving_distance(self, z, step=0.01):
        """
        Computes the comoving distance at given redshift(s).

        Args:
            z (float or list of floats): Redshift(s) at which to compute the comoving distance.
            step (float, optional): Step size for numerical integration. Defaults to 0.01.

        Returns:
            float or list of floats: Comoving distance(s) corresponding to the input redshift(s).
        """

        integrand = lambda z: speed_of_light/self.hubble_function(z)
        distance = [ trapz(integrand, 0, el, step ) for el in z]
        return distance

    def angular_diameter_distance(self,z, step=0.01):
        """
        Computes the angular diameter distance at given redshift(s).

        Args:
            z (float or list of floats): Redshift(s) at which to compute the angular diameter distance.
            step (float, optional): Step size for numerical integration. Defaults to 0.01.

        Returns:
            float or list of floats: Angular diameter distance(s) corresponding to the input redshift(s).
        """
        return self.comoving_distance(z, step=step)/(1+z)

    
    def luminosity_distance(self,z, step=0.01):
        """
        Computes the luminosity distance at given redshift(s).

        Args:
            z (float or list of floats): Redshift(s) at which to compute the luminosity distance.
            step (float, optional): Step size for numerical integration. Defaults to 0.01.

        Returns:
            float or list of floats: Luminosity distance(s) corresponding to the input redshift(s).
        """
        return self.comoving_distance(z, step=step)*(1+z)
