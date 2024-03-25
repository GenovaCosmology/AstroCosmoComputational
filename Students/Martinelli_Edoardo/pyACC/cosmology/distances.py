import numpy as np
from ..Calculus import integrate_f
c=299792 #km/s

# in hubble we have all the information of the model

def hubble(z, wde=-1, H0=67.0,omegaM=0.32,omegaLam=0.68,omegaGamma=0,omegaNu=0):
    '''
    hubble in units of km/s/Mpc
    
    WARNING: A non-zero value of curvature is given by the sum of the density parameters different from 1.

    Parameters
    ----------
    z : float or array
        Redshift
    wde : float (optional->default: LCDM)
        Dark energy zeldovich parameter
    H0 : float (optional->default: LCDM)
        Hubble constant
    omegaM : float (optional->default: LCDM)
        Matter and DM density
    omegaLam : float (optional->default: LCDM)
        Dark energy density
    omegaGamma : float (optional->default: LCDM)
        Radiation density
    omegaNu : float (optional->default: LCDM)
        Neutrino density
    ----------
    return : float or array
        Hubble parameter
    '''
    omegaK=1-(omegaLam+omegaM+omegaGamma+omegaNu)
    return np.sqrt(H0*H0*(omegaM*(1+z)**3+(omegaGamma+omegaNu)*(1+z)**4+omegaK*(1+z)**2+omegaLam*(1+z)**(3*(1+wde))))


                          
def comoving_distance(z, wde=-1, H0=67.0,omegaM=0.32, omegaLam=0.68,omegaGamma=0,omegaNu=0):
    '''
    comoving distance in Mpc

    WARNING: This function uses the function hubble above,
    please import it either from this module!
    
    Parameters
    ----------
    z : float or array
        Redshift
    wde : float (optional->default: LCDM)
        Dark energy zeldovich parameter
    H0 : float (optional->default: LCDM)
        Hubble constant
    omegaM : float (optional->default: LCDM)
        Matter and DM density
    omegaLam : float (optional->default: LCDM)
        Dark energy density
    omegaGamma : float (optional->default: LCDM)
        Radiation density
    omegaNu : float (optional->default: LCDM)
        Neutrino density
    ----------
    return : float or array
        Comoving distance
    '''
    omegaK=1-(omegaLam+omegaM+omegaGamma+omegaNu)
    if omegaK!=0:
        print('Curvature is not zero, use comoving_distance_transverse instead!')
        return None
    integrand = lambda x: c/hubble(x, wde, H0, omegaM, omegaLam, omegaGamma, omegaNu)
    if type(z) == float or type(z) == int:
        return integrate_f(integrand, 0, z)
    else:
        chiZ=np.zeros(z.size)
        for z_i,Z in enumerate(z):
            chiZ[z_i]=integrate_f(integrand, 0, Z)
        return chiZ


def comoving_distance_transverse(z, wde=-1, H0=67.0,omegaM=0.32, omegaLam=0.68,omegaGamma=0,omegaNu=0):
    '''
    comoving distance transverse in Mpc

    WARNING: This function uses the function comoving distance above,
    please import it either from this module!

    Parameters
    ----------
    z : float or array
        Redshift
    wde : float (optional->default: LCDM)
        Dark energy zeldovich parameter
    H0 : float (optional->default: LCDM)
        Hubble constant
    omegaM : float (optional->default: LCDM)
        Matter and DM density
    omegaLam : float (optional->default: LCDM)
        Dark energy density
    omegaGamma : float (optional->default: LCDM)
        Radiation density
    omegaNu : float (optional->default: LCDM)
        Neutrino density
    ----------
    return : float or array
        Comoving distance transverse
    '''
    omegaK=1-(omegaLam+omegaM+omegaGamma+omegaNu)
    dh=c/H0
    dc=comoving_distance(z, wde, H0, omegaM, omegaLam, omegaGamma, omegaNu)
    if omegaK>0:
        return (dh/np.sqrt(omegaK))*np.sinh((np.sqrt(omegaK)/dh)*dc)
    if omegaK==0:
        return dc
    if omegaK<0:
        return (dh/np.sqrt(-omegaK))*np.sin((np.sqrt(-omegaK)/dh)*dc)


def proper_distance(z, wde=-1, H0=67.0,omegaM=0.32, omegaLam=0.68,omegaGamma=0,omegaNu=0):
    '''
    proper distance in Mpc

    WARNING: This function uses the function comoving distance above,
    please import it either from this module!
    
    Parameters
    ----------
    z : float or array
        Redshift
    wde : float (optional->default: LCDM)
        Dark energy zeldovich parameter
    H0 : float (optional->default: LCDM)
        Hubble constant
    omegaM : float (optional->default: LCDM)
        Matter and DM density
    omegaLam : float (optional->default: LCDM)
        Dark energy density
    omegaGamma : float (optional->default: LCDM)
        Radiation density
    omegaNu : float (optional->default: LCDM)
        Neutrino density
    ----------
    return : float or array
        Proper distance
    '''
    return comoving_distance(z, wde, H0,omegaM, omegaLam,omegaGamma,omegaNu)*(1+z)**-1


def angular_diameter_distance(z, wde=-1, H0=67.0,omegaM=0.32, omegaLam=0.68,omegaGamma=0,omegaNu=0):
    '''
    angular diameter distance in Mpc

    WARNING: This function uses the function hubble and comoving distance above,
    please import them either from this module!
    
    Parameters
    ----------
    z : float or array
        Redshift
    wde : float (optional->default: LCDM)
        Dark energy zeldovich parameter
    H0 : float (optional->default: LCDM)
        Hubble constant
    omegaM : float (optional->default: LCDM)
        Matter and DM density
    omegaLam : float (optional->default: LCDM)
        Dark energy density
    omegaGamma : float (optional->default: LCDM)
        Radiation density
    omegaNu : float (optional->default: LCDM)
        Neutrino density
    ----------
    return : float or array
        Angular diameter distance
    '''
    return comoving_distance_transverse(z, wde, H0,omegaM, omegaLam,omegaGamma,omegaNu)*(1+z)**-1



def luminosity_distance(z, wde=-1, H0=67.0,omegaM=0.32, omegaLam=0.68,omegaGamma=0,omegaNu=0):
    '''
    luminosity distance in Mpc

    WARNING: This function uses the function hubble and comoving distance above,
    please import them either from this module!
    
    Parameters
    ----------
    z : float or array
        Redshift
    wde : float (optional->default: LCDM)
        Dark energy zeldovich parameter
    H0 : float (optional->default: LCDM)
        Hubble constant
    omegaM : float (optional->default: LCDM)
        Matter and DM density
    omegaLam : float (optional->default: LCDM)
        Dark energy density
    omegaGamma : float (optional->default: LCDM)
        Radiation density
    omegaNu : float (optional->default: LCDM)
        Neutrino density
    ----------
    return : float or array
        luminosity distance
    '''
    return comoving_distance_transverse(z, wde, H0,omegaM, omegaLam,omegaGamma,omegaNu)*(1+z)

def distance_modulus(z, wde=-1, H0=67.0,omegaM=0.32, omegaLam=0.68,omegaGamma=0,omegaNu=0):
    '''
    distance modulus

    WARNING: This function uses the function luminosity distance above,
    please import it either from this module!
    
    Parameters
    ----------
    z : float or array
        Redshift
    wde : float (optional->default: LCDM)
        Dark energy zeldovich parameter
    H0 : float (optional->default: LCDM)
        Hubble constant
    omegaM : float (optional->default: LCDM)
        Matter and DM density
    omegaLam : float (optional->default: LCDM)
        Dark energy density
    omegaGamma : float (optional->default: LCDM)
        Radiation density
    omegaNu : float (optional->default: LCDM)
        Neutrino density
    ----------
    return : float or array
        distance modulus
    '''
    return 5*np.log10(luminosity_distance(z, wde, H0,omegaM, omegaLam,omegaGamma,omegaNu))-5
    
    
