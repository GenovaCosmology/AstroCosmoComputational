import sys
sys.path.append("../")
from pyACC.Calculus import trapezi

c=3e5


def funz(z,H,omega_mat,omega_rad,w,H0):  
   return c/H(z,omega_mat,omega_rad,w,H0)

def Dc(H,z,omega_mat,omega_rad,w,H0):
    return trapezi(lambda z : funz(z,H,omega_mat,omega_rad,w,H0), 0, z, 1000) 