import numpy as np
from ..calculus import Integrate
from scipy.constants import c as speed_of_light

class CosmologicalDistances:
    """
    This class contains methods to compute the cosmological distances.
    """

    def __init__(self, hubble_function, *cosmo_pars):
        """
        Constructor of the class.

        Parameters
        ----------
        hubble_function : function
            Hubble function as a function of redshift.
        *cosmo_pars : list
            Additional arguments to be passed to the Hubble function.
        """

        self.hubble_function = hubble_function
        self.cosmo_pars = cosmo_pars

    def comoving_distance(self, z, *cosmo_pars, **integ_args):
        """
        Computes the comoving distance.

        Parameters
        ----------
        z : float or array_like
            Redshift.
        *cosmo_pars : list
            Additional arguments to be passed to the Hubble function.
        **integ_args : dict
            Additional arguments to be passed to the Integrate function.
        Returns
        -------
        float or array_like
            Comoving distance.
        """
        if len(cosmo_pars) !=0:
            integrand = lambda z: speed_of_light/1.e3 / self.hubble_function(z, *cosmo_pars)
        else:
            integrand = lambda z: speed_of_light/1.e3 / self.hubble_function(z, *self.cosmo_pars)
            
        return Integrate(integrand)("quadrature", 0.0, z, **integ_args)  

    def luminosity_distance(self, z, *cosmo_pars, **integ_args):
        """
        Computes the luminosity distance.

        Parameters
        ----------
        z : float or array_like
            Redshift.
        *cosmo_pars : list
            Additional arguments to be passed to the Hubble function.
        **integ_args : dict
            Additional arguments to be passed to the Integrate function.
        Returns
        -------
        float or array_like
            Luminosity distance.
        """
        return (1 + z) * self.comoving_distance(z, *cosmo_pars, **integ_args)
    
    def angular_diameter_distance(self, z, *cosmo_pars, **integ_args):
        """
        Computes the angular diameter distance.

        Parameters
        ----------
        z : float or array_like
            Redshift.
        *cosmo_pars : list
            Additional arguments to be passed to the Hubble function.
        **integ_args : dict
            Additional arguments to be passed to the Integrate function.
        Returns
        -------
        float or array_like
            Angular diameter distance.
        """
        return self.comoving_distance(z, *cosmo_pars, **integ_args) / (1 + z)
    
    def distance_modulus(self, z, *cosmo_pars, **integ_args):
        """
        Computes the distance modulus.

        Parameters
        ----------
        z : float or array_like
            Redshift.
        *cosmo_pars : list
            Additional arguments to be passed to the Hubble function.
        **integ_args : dict
            Additional arguments to be passed to the Integrate function.
        Returns
        -------
        float or array_like
            Distance modulus.
        """
        return 5 * np.log10(self.luminosity_distance(z, *cosmo_pars, **integ_args)) + 25