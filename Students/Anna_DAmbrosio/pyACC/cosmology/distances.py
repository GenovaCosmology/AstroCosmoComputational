import numpy as np
from ..Calculus import intgr

c = 299792.


def hubble(z, H_0=67.0, O_m=0.32, O_gam=0, O_nu=0, O_de=0.68, w_de=-1):
    '''
    Hubble in [km/sMpc]

    Parameters
    ----------
    z : float or array
        Redshift
    H_0 : float
        Hubble constant
    O_m : float
        Matter density
    O_gam : float
        Radiation density
    O_nu : float
        Neutrino density
    O_de : float
        Dark energy density
    w_de : float
        Dark energy Zeldovich parameter
    Return
    float or array
    Hubble
    '''
    O_k = 1- (O_m + O_gam + O_nu + O_de)
    return (H_0)*np.sqrt( (O_m)*(1+z)**3 + (O_gam+O_nu)*(1+z)**4 + O_de*(1+z)**(3*(1+w_de)) + O_k*(1+z)**2)



def comoving_distance(z,H_0=67.0, O_m=0.32, O_gam=0, O_nu=0, O_de=0.68, w_de=-1):
    '''
    Comiving distance

    Parameters
    ----------
    z : float or array
        Redshift
    H_0 : float
        Hubble constant
    O_m : float
        Matter density
    O_gam : float
        Radiation density
    O_nu : float
        Neutrino density
    O_de : float
        Dark energy density
    w_de : float
        Dark energy Zeldovich parameter
    Return
    float or array
    Comoving distance
    '''    
    integrand = lambda x: c/hubble(x, H_0, O_m, O_gam, O_nu, O_de, w_de)
    if type(z)==float or type(z)==int:
        return intgr(integrand, 0.0, z)[0]
    else:
        integral = np.zeros(z.size)
        for i,zi in enumerate(z):
            integral[i] = intgr(integrand, 0.0, zi)[0]
        return integral
    
def comoving_distance_transverse(z,H_0=67.0, O_m=0.32, O_gam=0, O_nu=0, O_de=0.68, w_de=-1):
    O_k = 1- (O_m + O_gam + O_nu-O_de)
    D_H = c/H_0
    D_C = comoving_distance(z, H_0, O_m, O_gam, O_nu, O_de, w_de)
    if O_k==0:
        return D_C
    if O_k>0:
        return D_H*(1/np.sqrt(O_k))*np.sinh(np.sqrt(O_k)*D_C/D_H)
    if O_k<0:
        return D_H*(1/np.sqrt(-O_k))*np.sin(np.sqrt(-O_k)*D_C/D_H)


def proper_distance(z,H_0=67.0, O_m=0.32, O_gam=0, O_nu=0, O_de=0.68, w_de=-1):
    '''
    Proper distance

    Parameters
    ----------
    z : float or array
        Redshift
    H_0 : float
        Hubble constant
    O_m : float
        Matter density
    O_gam : float
        Radiation density
    O_nu : float
        Neutrino density
    O_de : float
        Dark energy density
    w_de : float
        Dark energy Zeldovich parameter
    Return
    float or array
    Proper distance
    '''    
    return comoving_distance(z, H_0, O_m, O_gam, O_nu, O_de, w_de)/(1+z)

def luminosity_distance(z,H_0=67.0, O_m=0.32, O_gam=0, O_nu=0, O_de=0.68, w_de =-1):
    '''
    Luminosity distance

    Parameters
    ----------
    z : float or array
        Redshift
    H_0 : float
        Hubble constant
    O_m : float
        Matter density
    O_gam : float
        Radiation density
    O_nu : float
        Neutrino density
    O_de : float
        Dark energy density
    w_de : float
        Dark energy Zeldovich parameter
    Return
    float or array
    Luminosity distance
    '''    
    return (1+z)*comoving_distance_transverse(z, H_0, O_m, O_gam, O_nu, O_de, w_de)

def angular_diameter_distance(z,H_0=67.0, O_m=0.32, O_gam=0, O_nu=0, O_de=0.68, w_de=-1):
    '''
    Angular diameter distance

    Parameters
    ----------
    z : float or array
        Redshift
    H_0 : float
        Hubble constant
    O_m : float
        Matter density
    O_gam : float
        Radiation density
    O_nu : float
        Neutrino density
    O_de : float
        Dark energy density
    w_de : float
        Dark energy Zeldovich parameter
    Return
    float or array
    Angular diameter distance
    '''    
    return comoving_distance_transverse(z, H_0, O_m, O_gam, O_nu, O_de, w_de)/(1+z)

def distance_modulus(z,H_0=67.0, O_m=0.32, O_gam=0, O_nu=0, O_de=0.68, w_de=-1):
    return 5*np.log10(luminosity_distance(z, H_0, O_m, O_gam, O_nu, O_de, w_de))-5