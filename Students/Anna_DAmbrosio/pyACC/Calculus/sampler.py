import numpy as np
import scipy as sp
from .function import Function

def Rejection(target, x_in, x_f, N, type='flat'):
    pdf = Function(target)
    N_size = 1000000
    sample = []

    if type=='flat':
       rng = np.linspace(x_in, x_f, N_size)
       t_rng = pdf.Value(rng)
       t_rng_max = np.max(t_rng)
       
       def proposal(x):
           return t_rng_max*np.ones(x.size)
       
       for i in range(N):
            ent_guess = int(np.random.uniform(0,rng.size))
            alpha = t_rng[ent_guess]/t_rng_max
            u = np.random.uniform(0.0, 1.0)
            
            if u<=alpha:
                sample.append(rng[ent_guess])
                
    if type=='gaussian':
        func_target_mu = Function(lambda x: x*target(x))
        mu = func_target_mu.Integrate(-200,200)
        func_target_std = Function(lambda x: ((x-mu)**2)*target(x))
        std = np.sqrt(func_target_std.Integrate(-200,200))

        rng = sp.stats.truncnorm.rvs((x_in-mu)/std, (x_f-mu)/std, loc=mu, scale=std, size=N_size)
        t_rng = pdf.Value(rng)
        
        gauss = sp.stats.norm.pdf(rng,mu,std)

        c = np.max(t_rng/gauss)

        q = c*gauss

        proposal = lambda x: c*sp.stats.norm.pdf(x,mu,std)

        for i in range(N):
            ent_guess = int(np.random.uniform(0, rng.size))
            alpha = t_rng[ent_guess]/q[ent_guess]
            u = np.random.uniform(0.0, 1.0)
            
            if u<=alpha:
                sample.append(rng[ent_guess])

    return np.array(sample), (np.array(sample)).size/N, proposal

def Metropolis_Hastings(target, x0, N, cov=0, ds=0, type='flat'):
    '''
    Target must be normalized
    
    The proposal is a multivariate gaussian distribution with covariance cov and mean value the point where we are
    '''

    n_acc = 0
    sample = []
    pdf = Function(target)
    x_new = x0

    if type=='gaussian':
       
        for i in range(N):
            cand = np.random.default_rng().multivariate_normal(x_new,cov)   # generation of new value following a multivariate gaussian distribution
            ratio = pdf.Value(cand)/pdf.Value(x_new)    # acceptance ratio

            alpha = min(1,ratio)

            u = np.random.uniform(0.0, 1.0)

            if u<=alpha:
                x_new = cand
                n_acc = n_acc + 1 

            sample.append(x_new)
            
    if type=='flat':
       
    # ds is an array of the size of x0 containing the steps size for each variable

        for i in range(N):
            cand = x_new + ds*np.random.uniform(-1, 1, x_new.size)   # generation of new value following a multivariate gaussian distribution
              
            ratio = pdf.Value(cand)/pdf.Value(x_new)    # acceptance ratio

            alpha = min(1,ratio)

            u = np.random.uniform(0.0, 1.0)

            if u<=alpha:
                x_new = cand
                n_acc = n_acc + 1 
    
            sample.append(x_new)    # addition of the new value to the sample
    

    return np.array(sample), (n_acc)/N