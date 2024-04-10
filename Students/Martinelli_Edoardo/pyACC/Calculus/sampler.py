import numpy as np
from scipy.stats import norm
from scipy.stats import truncnorm
# Import function class
from .function import Funct

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
        for i in range(Nint):
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
        for i in range(Nint):
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
    - cov_mat: covariance matrix (numpy array)
    - x0: initial point (numpy array)
    - Nint: number of points in the sample (MC) (int)
    - ds: step size (array for multidimensional case) (float)
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
    
    sample = []
    nacc=0
    eff=0
    xnew=x0
    pdf=Funct(target_dist)
    if pdf_log==True:
        if type=='flat':
            for i in range(Nint):
                x_prime=xnew+ds*np.random.uniform(-1,1,xnew.size)
                acc_rate=np.min([0,pdf.Val(x_prime)-pdf.Val(xnew)])
                if np.log(np.random.uniform(1e-10,1))<acc_rate:
                    xnew=x_prime
                    nacc+=1
                sample.append(xnew)
                
        if type=='gauss':
            for i in range(Nint):
                x_prime=np.random.default_rng().multivariate_normal(xnew,cov_mat)
                acc_rate=np.min([0,pdf.Val(x_prime)-pdf.Val(xnew)])
                if np.log(np.random.uniform(1e-10,1))<acc_rate:
                    xnew=x_prime
                    nacc+=1
                sample.append(xnew)
    else:
        if type=='flat':
            for i in range(Nint):
                x_prime=xnew+ds*np.random.uniform(-1,1,xnew.size)
                acc_rate=np.min([1,pdf.Val(x_prime)/pdf.Val(xnew)])
                if np.random.uniform(0,1)<acc_rate:
                    xnew=x_prime
                    nacc+=1
                sample.append(xnew)

        if type=='gauss':
            for i in range(Nint):
                x_prime=np.random.default_rng().multivariate_normal(xnew,cov_mat)
                acc_rate=np.min([1,pdf.Val(x_prime)/pdf.Val(xnew)])
                if np.random.uniform(0,1)<acc_rate:
                    xnew=x_prime
                    nacc+=1
                sample.append(xnew)    
    eff=nacc/Nint
    return np.array(sample),eff