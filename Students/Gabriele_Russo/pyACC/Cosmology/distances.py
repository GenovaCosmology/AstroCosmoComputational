#Import of integration method
from ..Calculus import simpsons_rule

#Calculation of the comuving distance as integral made with trapezoid in dz of the Hubble function H(z) 
c = 3*10**5

#Defining Integrand Function
def Integrand_comoving_distance(z, H, Omega_mat, H_0):
        return c/H(z, Omega_mat, H_0)

#WARNING: trapezoid/simpson_rules needs a function as argument, you can't just put c/H since it's flot/function; 
#         this is why it has been necessary to define Integrand_comoving_distance
def comoving_distance(z, H, Omega_mat, H_0):
        return simpsons_rule(lambda z : Integrand_comoving_distance(z, H, Omega_mat, H_0), 0, z, 10000) #with lambda z : function, we are reducing the function only with z as variable

