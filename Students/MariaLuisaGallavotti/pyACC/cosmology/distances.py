from ..Calculus import integral

def integrand(z,H,OmegaM,H0):
    c=3*10**5
    return c/H(z,OmegaM,H0)

def Dc(H,z,OmegaM,H0):
    return integral(lambda t : integrand(t,H,OmegaM,H0),0,z,1000)

#integral= lambda z : 1/H(z,OmegaM,H0)
#tutto quello che c'è dopo lambda è variabile, mentre gli altri parametri sono fissi