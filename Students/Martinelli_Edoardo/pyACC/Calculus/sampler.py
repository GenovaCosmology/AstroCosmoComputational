import numpy as np
from scipy.stats import norm
from scipy.stats import truncnorm
from tqdm import tqdm
# Import function class
from .function import Funct
from inspect import signature 

def gaussian(x, mu, sig):
    return 1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.0) / 2)

def Rejection_sampling(target_dist,xin,xfin,Nint,ext_int=200,type='flat'):
    '''
    This function generates a sample of points from a target distribution using the Rejection Sampling algorithm.

    Parametrs:
    -----------
    - target_dist: target distribution (must be normalized) (function lambda)
    - xin: initial value of the range (int or float)
    - xfin: final value of the range (int or float)
    - Nint: number of points in the sample (MC) (int)
    - ext_int: extension of the range of integration for numerical porpuses (int or float)
    - type: type of target distribution ('flat' or 'gauss') (str)
            flat: flat proposal
            gauss: gaussian proposal
    Returns:
    -----------
    - sample: sample of points from the target distribution (numpy array)
    - eff: efficiency of the algorithm (float)
    - window: window function (function lambda)
    '''
    if callable(target_dist)==False:
        raise ValueError('Target distribution must be a function (lambda).')
    sample=[]
    eff=0
    Nsize=1000000
    PDF=Funct(target_dist)
    

    if type=='flat':
        x=np.linspace(xin,xfin,Nsize)
        pdf_eval=PDF.Val(x)
        val_max=np.max(pdf_eval)
        def window(y):
            return np.ones(y.size)*val_max
        for i in tqdm(range(Nint),desc='Rejection Sampling: '):
            i_guess=int(np.random.uniform(0,x.size))
            if pdf_eval[i_guess] >= np.random.uniform(0,val_max):
                sample.append(x[i_guess])
        
    if type=='gauss':
        func_mu=Funct(lambda x: x*target_dist(x))
        mu=func_mu.Int(-ext_int,ext_int) # ext_int for numerical porpuses
        func_sig=Funct(lambda x: target_dist(x)*(x-mu)**2)
        sig=np.sqrt(func_sig.Int(-ext_int,ext_int)) # ext_int for numerical porpuses

        x=truncnorm.rvs((xin-mu)/sig, (xfin-mu)/sig, loc=mu, scale=sig, size=Nsize)
        pdf_eval=PDF.Val(x)

        #mu=np.trapz(x*pdf_eval,x)
        #sig=np.sqrt(np.trapz(pdf_eval*(x-mu)**2,x))
        gauss=gaussian(x,mu,sig)
        c=np.max(np.abs(pdf_eval/gauss))
        proposal=c*gauss
        window=lambda y: c*gaussian(y,mu,sig)
        for i in tqdm(range(Nint),desc='Rejection Sampling: '):
            i_guess=int(np.random.uniform(0,x.size))
            if pdf_eval[i_guess] >= np.random.uniform(0,proposal[i_guess]):
                sample.append(x[i_guess])
            

    sample=np.array(sample)
    eff=sample.size/Nint
    return sample,eff,window


def Metropolis_Hastings(target_dist,x0,Nint,ds=0,cov_mat=0,type='flat',pdf_log=False):
    '''
    This function generates a sample of points from a target distribution using the Metropolis-Hastings algorithm.

    Parametrs:
    -----------
    - target_dist: target distribution (function lambda)
    - cov_mat: covariance matrix (numpy array) for the gaussian proposal only
    - x0: initial point (numpy array)
    - Nint: number of points in the sample (MC) (int)
    - ds: step size (array for multidimensional case) (float) for the flat proposal only
    - type: type of target distribution ('flat' or 'gauss') (str)
            flat: flat proposal
            gauss: gaussian proposal
    - pdf_log: if the target_dist is log, it extract from -inf to 0 the acceptance rate
    Returns:
    -----------
    - sample: sample of points from the target distribution (numpy array)
              It will be an array whose entries are x0.size arrays
    - eff: efficiency of the algorithm (float)
    '''
    if callable(target_dist)==False:
        raise ValueError('Target distribution must be a function (lambda).')
    print('-------------------------------')
    print('Metropolis-Hastings algorithm: ')
    sig = signature(target_dist)
    args = []
    for arg in sig.parameters.values():
        args.append(str(arg))
    print('You have inserted a distribution with these arguments: ', args)
    print('-------------------------------')
    print('The starting point in parameter space is: ', x0)
    print('-------------------------------')

    sample = []
    nacc=0
    xnew=x0
    pdf=Funct(target_dist)

    if type=='gauss':
        for i in tqdm(range(Nint),desc='M-H, Gaussian proposal: '):
            if pdf_log==True:
                x_prime=np.random.default_rng().multivariate_normal(xnew,cov_mat)
                if pdf.Val(x_prime)==0:
                    acc_rate=-np.inf
                else:
                    acc_rate=np.min([0,pdf.Val(x_prime)-pdf.Val(xnew)])
                u=np.log(np.random.uniform(1e-10,1))
            else:
                x_prime=np.random.default_rng().multivariate_normal(xnew,cov_mat)
                acc_rate=np.min([1,pdf.Val(x_prime)/pdf.Val(xnew)])
                u=np.random.uniform(0,1)

            if u<acc_rate:
                    xnew=x_prime
                    nacc+=1
            sample.append(xnew)

    if type=='flat':
        for i in tqdm(range(Nint),desc='M-H, Flat proposal: '):
            if pdf_log==True:
                x_prime=xnew+ds*np.random.uniform(-1,1,xnew.size)
                if pdf.Val(x_prime)==0:
                    acc_rate=-np.inf
                else:
                    acc_rate=np.min([0,pdf.Val(x_prime)-pdf.Val(xnew)])
                u=np.log(np.random.uniform(1e-10,1))
            else:
                x_prime=xnew+ds*np.random.uniform(-1,1,xnew.size)
                acc_rate=np.min([1,pdf.Val(x_prime)/pdf.Val(xnew)])
                u=np.random.uniform(0,1)

            if u<acc_rate:
                    xnew=x_prime
                    nacc+=1
            sample.append(xnew)
       
    return np.array(sample),nacc/Nint
