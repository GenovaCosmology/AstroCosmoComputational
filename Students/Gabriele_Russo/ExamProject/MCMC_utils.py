'''
A simple code to run MCMC for SH0ES results.
Y.S.Murakami September 2021 @ JHU
'''
import numpy as np
from scipy import linalg
from astropy.io import fits
try:
    import emcee
except Exception:
    print('emcee is required to run this code. Try \'pip install emcee\'.')

def log_prior(theta,priors):
    '''
    log-prior for the parameters (uniform):
    - theta:
        array, parameters to be tested
    - priors:
        tuple, (mu,halfwidth) for each parameter, 
        i.e. the central value and half-width of the uniform prior.
    '''
    mu,halfwidth = priors
    for i in range(len(theta)):
        if theta[i]>mu[i]+halfwidth[i] or theta[i]<mu[i]-halfwidth[i]:
            return -np.inf # log(0), i.e. impossible under our prior
    return 1 # log(1) = 0, i.e. uniform prior,
    # it returns 1 because confronting differente log_priors shifted by 1 is as they weren't shifted

def log_probability(theta,Y,L,C,C_inv_cho,priors):
    '''
    log-probability (posterior) that governs the rate of acceptance for given proposed parameter:
    - theta:
        array, parameters to be tested
    - Y:
        array, data
    - L:
        multi-dimensional array, matrix
    - C:
        multi-dimensional array, matrix
    - C_inv_cho:
        multi-dimensional array, matrix, i.e. C_inv, but with
        Cholesky decomposition applied for faster calculation
    - priors:
        tuple, (mu,halfwidth) for each parameter, 
        i.e. the central value and half-width of the uniform prior.
    '''
    lp = log_prior(theta,priors) # compute log-prior
    if not np.isfinite(lp): # if log-prior is -inf (is not finite), return -inf
        return -np.inf
    ll = log_likelihood(theta,Y,L,C,C_inv_cho) # compute log-likelihood
    if not np.isfinite(ll): # if log-likelihood is -inf (is not finite), return -inf
        return -np.inf
    return lp + ll # return log-posterior = log-prior + log-likelihood

'''
OBS.:
Cholesky decomposition is a numerical technique to factorize a symmetric, 
positive-definite matrix into a product of a lower triangular matrix and 
its transpose, which can be used to solve linear systems more stably.
'''

# log-likelihood function: remind that for data normally distributed, 
# the likelihood is proportional to exp(-0.5 * chi2) 
def log_likelihood(theta,Y,L,C,C_inv_cho):
    res = Y-np.dot(theta,L)
    chi2 = np.dot(res,np.dot(C_inv_cho,res))
    return -0.5*chi2  
# normalization is constant so can be omitted since would be
# added (we're working with log) 

def run_MCMC(nwalkers,chain,data_paths,lstsq_results,
             prior_width_ratio,outpath=None,contd=False):
    '''
    a wrapper file;
    Parameter:
    - nwalkers:
        int, number of walkers, i.e. number of chains
    - chain:
        int, length of chain for each walker (total number of samples = nwalkers * chain), 
        i.e. how many steps to take for each chain
    - data_paths:
        list, file paths for Y, L, C matrices
    - lstsq_results:
        list, best-fit results from least-square fit
    - prior_width_ratio:
        float, ratio between least-square fit standard deviation to the half-width of uniform prior
    - outpath:
        str, output file path
    - contd:
        bool, set True to resume sampling (for specific outpath)
    '''
    if outpath is None:
        outpath = './results.h5'
    # load data
    Y_fits_path, L_fits_path, C_fits_path = data_paths
    Y = fits.open(Y_fits_path)[0].data
    L = fits.open(L_fits_path)[0].data
    C = fits.open(C_fits_path)[0].data
    
    # prepare inverse of C-matrix
    C_inv_cho = linalg.cho_solve(linalg.cho_factor(C),np.identity(C.shape[0]))

    # priors
    q_lstsq, sigma_lstsq = lstsq_results
    mu_list = q_lstsq
    width_list = sigma_lstsq * prior_width_ratio
    priors = [mu_list,width_list]
    
    # initial guess array for each walker: 
    x0 = np.random.uniform(mu_list-width_list,
                           mu_list+width_list,
                           size=(nwalkers,len(mu_list)))
    nwalkers, ndim = x0.shape # shape of x0 is set to be (# of chains, # of parameters)
    
    # save file
    backend = emcee.backends.HDFBackend(outpath)
    if contd:
        print(f'initial size: {backend.iteration}',flush=True)
    else:
        backend.reset(nwalkers, ndim)
    
    # initialize sampler, run MCMC

    # initialize sampler with the number of walkers, the number of parameters,
    # and the log-probability function defined above 
    sampler = emcee.EnsembleSampler(nwalkers, 
                                    ndim, 
                                    log_probability,
                                    args = [Y,L,C,C_inv_cho,priors],
                                    backend = backend)
    
    '''
        The arguments of sampler.run_mcmc are:
        - x0: initial guess array for each walker
          or None to start from the current position if contd=True 
        - chain: length of chain for each walker
        - progress: set True to show progress bar
        - skip_initial_state_check: set True to skip initial state check
          (useful when resuming sampling with contd=True)
    '''
    if contd:
        sampler.run_mcmc(None, chain, progress=True,skip_initial_state_check=True);
        print(f'final size: {backend.iteration}',flush=True)
    else:
        sampler.run_mcmc(x0, chain, progress=True,skip_initial_state_check=True);