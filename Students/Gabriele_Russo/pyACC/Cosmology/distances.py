#Import of integration method
from ..Calculus import simpsons_rule
import numpy as np

#Calculation of the comuving distance as integral made with trapezoid in dz of the Hubble function H(z) 
c = 3*10**5

#Defining Integrand Function
def Integrand_comoving_distance(z, H, Omega_mat, H_0):
        return c/H(z, Omega_mat, H_0)

#WARNING: trapezoid/simpson_rules needs a function as argument, you can't just put c/H since it's flot/function; 
#         this is why it has been necessary to define Integrand_comoving_distance
def comoving_distance(z, H, Omega_mat, H_0):
        return simpsons_rule(lambda z : Integrand_comoving_distance(z, H, Omega_mat, H_0), 0, z, 10000) #with lambda z : function, we are reducing the function only with z as variable

#*******************#
#Half Course Project#
#*******************#
#Class for computation of Cosmologiacal Distances
class CosmologicalDistances:
        #Constructor of the objects of the class
        def __init__(self, hubble_function, omega_m, H0): 
                self.hubble_function = hubble_function
                self.omega_m = omega_m
                self.H0 = H0
        
        #*************#
        #Class Methods#
        #*************#
        def distance_modulus(self, z):
                '''
                Method to compute distances modulus:
                Parameters
                ----------
                z : float
                Redshift.

                Returns
                -------
                float
                        m, distance modulus at redshift z.
                '''
                #Computing comuving distances
                comoving_distances = comoving_distance(z, self.hubble_function, self.omega_m, self.H0)

                #Computing distances modulus
                m = 5*np.log10(comoving_distances*1e5) + 25 

                return m
        
        def luminosity_distance(self, z):
