import numpy as np
from ..calculus import integral,trapezoid
import scipy
import scipy.integrate as integrate

c = 3e5   #km/s
G=6.67e-11

# lambdaCDM parameters are the default parameters of all the functions below


#Dc = Comoving distances
#goes from now (z=0) to z
def Dc(z_f , H0=67.7, omegam=0.319, omegar=0,w_de= -1):
    integ = lambda z : c/ ( H0* np.sqrt ( omegam*(1+z)**3 + omegar*(1+z)**4 + (1-omegam-omegar)*np.power(1+z,3*(1+w_de))))
    return integral(integ,0,z_f)
    #return integrate.quad(integ,0,z_f)


#Dm_t =  Comoving distance (transverse) = Dc in a flat space
def Dm_t(z_f , H0=67.7, omegam=0.319, omegar=0,w_de= -1,omegak=0):
    integ = lambda z : c/ ( H0* np.sqrt ( omegam*(1+z)**3 + omegar*(1+z)**4 + omegak*(1+z)**2+ (1-omegam-omegar)*np.power(1+z,3*(1+w_de))))
    temp = integral(integ,0,z_f)
    if omegak==0:
        return temp
    elif omegak>0:
        return (c/(H0*np.sqrt(omegak)))*np.sinh(np.sqrt(omegak)*temp/(c/H0))
    else:
        return (c/(H0*np.sqrt(-omegak)))*np.sin(np.sqrt(-omegak)*temp/(c/H0))


#Dp = proper distances= a(t)Dm_t = angular diameter distance


#Dl = luminosity distance = (1+z)*Dm_t
def Dl(z_f , H0=67.7, omegam=0.319, omegar=0,w_de= -1,omegak=0):
    integ = lambda z : c/ ( H0* np.sqrt ( omegam*(1+z)**3 + omegar*(1+z)**4 + omegak*(1+z)**2 +(1-omegam-omegar)*np.power(1+z,3*(1+w_de))))
    temp = integral(integ,0,z_f)
    if omegak==0:
        return temp*(1+z_f)
    elif omegak>0:
        return (1+z_f)*(c/(H0*np.sqrt(omegak)))*np.sinh(np.sqrt(omegak)*temp/(c/H0))
    else:
        return (1+z_f)*(c/(H0*np.sqrt(-omegak)))*np.sin(np.sqrt(-omegak)*temp/(c/H0))


#Da = angular diameter distance = Dm_t/(1+z)
def Da(z_f , H0=67.7, omegam=0.319, omegar=0,w_de= -1,omegak=0):
    integ = lambda z : c/ ( H0* np.sqrt ( omegam*(1+z)**3 + omegar*(1+z)**4 + omegak*(1+z)**2 +(1-omegam-omegar)*np.power(1+z,3*(1+w_de))))
    temp = integral(integ,0,z_f)
    if omegak==0:
        return temp/(1+z_f)
    elif omegak>0:
        return (c/(H0*np.sqrt(omegak)))*np.sinh(np.sqrt(omegak)*temp/(c/H0))/(1+z_f)
    else:
        return (c/(H0*np.sqrt(-omegak)))*np.sin(np.sqrt(-omegak)*temp/(c/H0))/(1+z_f)


#Dm = distance modulus = magnitudine apparente - magnitudine assoluta = m - M
def Dm(z_f , H0=67.7, omegam=0.319, omegar=0,w_de=-1):
    integ = lambda z : c/ ( H0* np.sqrt ( omegam*(1+z)**3 + omegar*(1+z)**4 + (1-omegam-omegar)*np.power(1+z,3*(1+w_de))))
    temp  = (1+z_f)*integral(integ,0,z_f)
    return 5*np.log10(temp) - 5
