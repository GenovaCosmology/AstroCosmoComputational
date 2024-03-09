from ..calculus import integral

def integrand(z,H,OmegaM,OmegaR,w,H0):
    c=3*10**5 #km/s
    return c/H(z,OmegaM,OmegaR,w,H0)

def Dc(H,z,OmegaM,OmegaR,w,H0):
    return integral(lambda t : integrand(t,H,OmegaM,OmegaR,w,H0),0,z,0.000001)

#integral= lambda z : 1/H(z,OmegaM,OmegaR,w,H0)
#tutto quello che c'è dopo lambda è variabile, mentre gli altri parametri sono fissi