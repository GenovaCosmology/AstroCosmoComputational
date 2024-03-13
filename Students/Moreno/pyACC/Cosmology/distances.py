import sys
sys.path.append("/home/git/AstroCosmoComputational/Students/Moreno")
from pyACC.Calculus import trapezoid
c= 3*10**5



def Int_Dc(z, H, Omega_mat, H0):
    
    return c/H(z, Omega_mat, H0)

def Dc(H, z, Omega_mat, H0, delta_z):
    return trapezoid(lambda z : Int_Dc(z, H, Omega_mat, H0),0,z,delta_z)
