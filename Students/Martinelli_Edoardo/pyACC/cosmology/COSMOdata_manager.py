import numpy as np
from numpy.linalg import inv
from numpy.linalg import multi_dot
from ..Calculus import *
from ..clustering import *
from tqdm import tqdm
from inspect import signature 
c=299792 #km/s

rd=147.09 #Mpc, from Planck 2018

def dataDESImanager(data):
    '''
    data must be a numpy matrix (Nx7) with the following columns:
    0: redshift
    1: DH (Hubble distance)
    2: deltaDH (error on DH)
    3: DM (comoving distance)
    4: deltaDM (error on DM)
    5: DV (volume distance)
    6: deltaDV (error on DV)
    The function is going to compute DV from DH and DM, and the corresponding error,
    if DV is not provided (DV=deltaDV=0); if DV is provided, set 0 the corresponding
    element in columns 1,2,3,4.
    ATTENTION--> We assume NO correlation between the errors on DH and DM.
    ------------------------------------------------
    return: 
    ->numpy matrix with the following columns:
        0: redshift
        1: DV (volume distance)
    ->covariance matrix of DV (diagonal because we assume no correlation between DH and DM)
    '''
    data=np.array(data)
    z_array=data[:,0]
    DV_array=[]
    deltaDV_array=[]
    for i in tqdm(range(data.shape[0]),desc='Computing data: '):
        dv=(data[i,0]*(data[i,1]*data[i,3]**2))**(1./3)
        if data[i,3]==0 and data[i,1]==0:
            Ddv=0.0
        else:
            Ddv=((2./3)*data[i,3]**(-1./3)*data[i,1]**(1./3)*data[i,4]+(1./3)*data[i,3]**(2./3)*data[i,1]**(-2./3)*data[i,2])*data[i,0]**(1./3)
        DV_array.append(data[i,5]+dv)
        deltaDV_array.append(data[i,6]+Ddv)
    
    return np.column_stack([z_array,np.array(DV_array)]), np.diag(np.array(deltaDV_array)**2)

def get_logLIKELIHOOD(data, func_dist, sigma=None, type='DESI'):
    
    '''
    It return the log likelihood distribution of the data given the model.
    It will be a function of the same parameters of the function func_dist.

    Parameters:
    data: numpy matrix shape type 'DESI' (see function dataDESImanager) or 'SNIa' (Nx2)
          for 'SNIa' the columns are: 0: redshift, 1: mu (distance modulus)
    func_dist: lambda function with first argument the redshift array and the other arguments the parameters of the model
               function that returns the distance (volume distance for DESI, distance modulus for SNIa)
    sigma: ONLY for 'SNIa' type (NxN np matrix) (default ('DESI'): None)
           covariance matrix of the distance measurements
    type: string (default: 'DESI')
          type of data ('DESI' or 'SNIa')
    ------------------------------------------------
    Return:
    log-likelihood distribution (Lambda function)
    ------------------------------------------------
    '''
    # here we manage the structure of DESI data to match the structure of SNIa data
    if type=='DESI':
        data, sigma = dataDESImanager(data)
    
    # redshift and distances arrays
    z = data[:,0]
    D = data[:,1]
    n = z.shape[0]
    Cov_in=np.array(np.linalg.inv(sigma))

    # let's play with function arguments
    sig = signature(func_dist)
    args = []
    for arg in sig.parameters.values():
        args.append(str(arg))
    args.pop(0) # remove the first argument (redshift)
    print('Arguments of the LH function: ',args)

    # let's define the log-likelihood function
    if type=='SNIa':
        LH = lambda *args: 0.5*(-multi_dot([D-func_dist(z,*args),Cov_in,D-func_dist(z,*args)]) + ((multi_dot([np.ones(n),Cov_in,D-func_dist(z,*args)]))**2)/(multi_dot([np.ones(n),Cov_in,np.ones(n)])))
    else:
        LH = lambda *args: -0.5*multi_dot([D-func_dist(z,*args),Cov_in,D-func_dist(z,*args)]) * prior(*args)
    
    return LH



def get_logLIKELIHOOD_flatPRIOR(data, func_dist, infs, sups, sigma=None, type='DESI'):
    
    '''
    It return the log likelihood distribution of the data given the model with a flat prior.
    It will be a function of the same parameters of the function func_dist.

    Parameters:
    data: numpy matrix shape type 'DESI' (see function dataDESImanager) or 'SNIa' (Nx2)
          for 'SNIa' the columns are: 0: redshift, 1: mu (distance modulus)
    func_dist: lambda function with first argument the redshift array and the other arguments the parameters of the model
               function that returns the distance (volume distance for DESI, distance modulus for SNIa)
    infs: list of lower bounds for the parameters
    sups: list of upper bounds for the parameters
    sigma: ONLY for 'SNIa' type (NxN np matrix) (default ('DESI'): None)
           covariance matrix of the distance measurements
    type: string (default: 'DESI')
          type of data ('DESI' or 'SNIa')
    ------------------------------------------------
    Return:
    log-likelihood distribution * prior (Lambda function)
    ------------------------------------------------
    '''
    # here we manage the structure of DESI data to match the structure of SNIa data
    if type=='DESI':
        data, sigma = dataDESImanager(data)
    
    # redshift and distances arrays
    z = data[:,0]
    D = data[:,1]
    n = z.shape[0]
    Cov_in=np.array(np.linalg.inv(sigma))

    # let's play with function arguments
    sig = signature(func_dist)
    args = []
    for arg in sig.parameters.values():
        args.append(str(arg))
    args.pop(0) # remove the first argument (redshift)
    print('Arguments of the LH function: ',args)
    
    flat_prior = lambda *args: 1.0 if ((np.array(args)>infs).all() and (np.array(args)<sups).all()) else 0.0

    # let's define the log-likelihood function
    if type=='SNIa':
        LH = lambda *args: (0.5*(-multi_dot([D-func_dist(z,*args),Cov_in,D-func_dist(z,*args)]) + ((multi_dot([np.ones(n),Cov_in,D-func_dist(z,*args)]))**2)/(multi_dot([np.ones(n),Cov_in,np.ones(n)]))))* flat_prior(*args)
    else:
        LH = lambda *args: (-0.5*multi_dot([D-func_dist(z,*args),Cov_in,D-func_dist(z,*args)])) * flat_prior(*args)
    
    return LH