from ..Calculus.integrate import integrale_trapezio

c=3*10**5 #km/s 

def Integrand(z, H, omegaM, H0):
    return c/H(z, omegaM, H0)

def Dc(H, z, omegaM, H0):
    return integrale_trapezio(lambda t: Integrand(t, H, omegaM, H0), 0, z, (z-0)/1000)