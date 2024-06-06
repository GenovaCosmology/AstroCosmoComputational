import numpy as np
from ..Calculus import *
c=299792 #km/s
rd=147.09 #Mpc, drag scale from Planck 2018

# in hubble we have all the information of the model

def hubble(z, wde=-1, H0=67.0,omegaM=0.32,omegaLam=0.68,omegaGamma=0,omegaNu=0,w0=0,wa=0):
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
    w0 : float (optional->default: 0)
        Dark energy equation of state parameter in CPL parametrization
    wa : float (optional->default: 0)
        Dark energy equation of state parameter in CPL parametrization
    ----------
    return : float or array
        Hubble parameter
    '''
    omegaK=1-(omegaLam+omegaM+omegaGamma+omegaNu)
    if w0==0 and wa==0:
        return np.sqrt(H0*H0*(omegaM*(1+z)**3+(omegaGamma+omegaNu)*(1+z)**4+omegaK*(1+z)**2+omegaLam*(1+z)**(3*(1+wde))))
    else:
        # CPL parametrization
        return np.sqrt(H0*H0*(omegaM*(1+z)**3+(omegaGamma+omegaNu)*(1+z)**4+omegaK*(1+z)**2+omegaLam*(1+z)**(3*(1+w0+wa))*np.exp(-3*wa*(z/(1+z)))))
    
       
def comoving_distance(z, wde=-1, H0=67.0,omegaM=0.32, omegaLam=0.68,omegaGamma=0,omegaNu=0,w0=0,wa=0):
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
    w0 : float (optional->default: 0)
        Dark energy equation of state parameter in CPL parametrization
    wa : float (optional->default: 0)
        Dark energy equation of state parameter in CPL parametrization
    ----------
    return : float or array
        Comoving distance
    '''
    omegaK=1-(omegaLam+omegaM+omegaGamma+omegaNu)
    # if omegaK!=0:
    #     print('Curvature is not zero', omegaK, ', use comoving_distance_transverse instead!')
    #     return None
    integrand = lambda x: c/hubble(x, wde, H0, omegaM, omegaLam, omegaGamma, omegaNu, w0, wa)
    if type(z) == float or type(z) == int:
        return integrate_f(integrand, 0, z)
    else:
        chiZ=np.zeros(z.size)
        for z_i,Z in enumerate(z):
            chiZ[z_i]=integrate_f(integrand, 0, Z)
        return chiZ


def comoving_distance_transverse(z, wde=-1, H0=67.0,omegaM=0.32, omegaLam=0.68,omegaGamma=0,omegaNu=0,w0=0,wa=0):
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
    w0 : float (optional->default: 0)
        Dark energy equation of state parameter in CPL parametrization
    wa : float (optional->default: 0)
        Dark energy equation of state parameter in CPL parametrization
    ----------
    return : float or array
        Comoving distance transverse
    '''
    omegaK=1-(omegaLam+omegaM+omegaGamma+omegaNu)
    dh=c/H0
    dc=comoving_distance(z, wde, H0, omegaM, omegaLam, omegaGamma, omegaNu, w0, wa)
    if omegaK>0:
        return (dh/np.sqrt(omegaK))*np.sinh((np.sqrt(omegaK)/dh)*dc)
    if omegaK==0:
        return dc
    if omegaK<0:
        return (dh/np.sqrt(-omegaK))*np.sin((np.sqrt(-omegaK)/dh)*dc)


def proper_distance(z, wde=-1, H0=67.0,omegaM=0.32, omegaLam=0.68,omegaGamma=0,omegaNu=0,w0=0,wa=0):
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
    w0 : float (optional->default: 0)
        Dark energy equation of state parameter in CPL parametrization
    wa : float (optional->default: 0)
        Dark energy equation of state parameter in CPL parametrization
    ----------
    return : float or array
        Proper distance
    '''
    return comoving_distance(z, wde, H0,omegaM, omegaLam,omegaGamma,omegaNu, w0, wa)*(1+z)**-1


def angular_diameter_distance(z, wde=-1, H0=67.0,omegaM=0.32, omegaLam=0.68,omegaGamma=0,omegaNu=0,w0=0,wa=0):
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
    w0 : float (optional->default: 0)
        Dark energy equation of state parameter in CPL parametrization
    wa : float (optional->default: 0)
        Dark energy equation of state parameter in CPL parametrization
    ----------
    return : float or array
        Angular diameter distance
    '''
    return comoving_distance_transverse(z, wde, H0,omegaM, omegaLam,omegaGamma,omegaNu, w0, wa)*(1+z)**-1



def luminosity_distance(z, wde=-1, H0=67.0,omegaM=0.32, omegaLam=0.68,omegaGamma=0,omegaNu=0,w0=0,wa=0):
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
    w0 : float (optional->default: 0)
        Dark energy equation of state parameter in CPL parametrization
    wa : float (optional->default: 0)
        Dark energy equation of state parameter in CPL parametrization
    ----------
    return : float or array
        luminosity distance
    '''
    return comoving_distance_transverse(z, wde, H0,omegaM, omegaLam,omegaGamma,omegaNu, w0, wa)*(1+z)

def distance_modulus(z, wde=-1, H0=67.0,omegaM=0.32, omegaLam=0.68,omegaGamma=0,omegaNu=0,w0=0,wa=0):
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
    w0 : float (optional->default: 0)
        Dark energy equation of state parameter in CPL parametrization
    wa : float (optional->default: 0)
        Dark energy equation of state parameter in CPL parametrization
    ----------
    return : float or array
        distance modulus
    '''
    return 5*np.log10(luminosity_distance(z, wde, H0,omegaM, omegaLam,omegaGamma,omegaNu, w0, wa))-5
    
    
# distance volume DV for DESI implementation

def distance_volume_over_rd(z, wde=-1, H0=67.0,omegaM=0.32, omegaLam=0.68,omegaGamma=0,omegaNu=0,w0=0,wa=0):
    
    '''
    DV distance in Mpc

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
    w0 : float (optional->default: 0)
        Dark energy equation of state parameter in CPL parametrization
    wa : float (optional->default: 0)
        Dark energy equation of state parameter in CPL parametrization
    ----------
    return : float or array
        DV distance
    '''
    H=hubble(z, wde, H0, omegaM, omegaLam, omegaGamma, omegaNu, w0, wa)
    DCt=comoving_distance_transverse(z, wde, H0, omegaM, omegaLam, omegaGamma, omegaNu, w0, wa)
    return ((z*((c/H)*DCt**2))**(1./3))/rd