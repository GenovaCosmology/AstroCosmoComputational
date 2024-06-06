import numpy as np
from ..Calculus import *
from ..clustering import *
from inspect import signature 

c=299792 #km/s

def dataDESImanager(data):
    '''
    Parameters
    ----------
    data : array
        Data from galaxy survey
        must have the following columns: 'z', 'D_H', 'deltaD_H', 'D_M', 'deltaD_M', 'D_V', 'deltaD_V'
        if a quantity is not provided it must be set to 0
        it can be given eather D_H and D_M or D_V
        we assume no correlation between datas
    Returns
    -------
    data_DESI : array
        Data from galaxy survey
        with the following columns: 'z', 'D_V'
        if the data given are not directly D_V, it will be calculated through D_H and D_M
        in this case the errors will be propagated
    cov: array
        Covariance matrix
    '''
    z_array = data[:,0]
    D_V_array = []
    deltaD_V_array = []

    for i in range(data.shape[0]):
        z = data[i,0]
        D_H = data[i,1]
        deltaD_H = data[i,2]
        D_M = data[i,3]
        deltaD_M = data[i,4]
        
        D_V = (z*D_M**2*D_H)**(1./3)
        if D_H==0 and D_M==0:
            deltaD_V = 0.0
        else:
            deltaD_V = z**(1./3)*((2./3)*D_M**(-1./3)*D_H**(1./3)*deltaD_M + (1./3)*D_M**(2./3)*D_H**(-2./3)*deltaD_H)
        
        D_V_array.append(data[i,5] + D_V)
        deltaD_V_array.append(data[i,6] + deltaD_V)

    cov = np.diag((np.array(deltaD_V_array))**2)

    return np.column_stack([z_array,np.array(D_V_array)]), cov


def logL(data, func_dist, sigma=None, type='DESI'):
    '''
    Parameters
    ----------
    data : array
        Data from galaxy survey
        must have the following columns: 'z', 'D_H', 'deltaD_H', 'D_M', 'deltaD_M', 'D_V', 'deltaD_V' if type=='DESI'
        must have the following columns: 'z', 'mu' and the covariance matrix must be given as sigma if type=='SNIa'
    func_dist : function
        Function to calculate the distance,
        must be a lambda function that has as variable the redshift, the paramaeters that we want to estimate;
        the other parameters must be fixed
        (can be any distance function, distance modulus or distance volume);
        must be defines in the notebook
    sigma : array
        Covariance matrix
    type : string
        Type of data
        'SNIa' for supernovae
        'DESI' for DESI experiment data
    Returns
    -------
    logL : float
        Log likelihood
    '''

    if type == 'DESI':
        data, sigma = dataDESImanager(data)

    z = data[:,0]
    D = data[:,1]
    n = z.shape[0]

    ones = np.ones(n)
    args = []
    args_sign = signature(func_dist)
    for param in args_sign.parameters.values():
        args.append(str(param))
    args.pop(0)

    print('Arguments of LH: ', args)

    sigma_inv = np.array(np.linalg.inv(sigma))
    if type == 'SNIa':
        LH = lambda *args: -(1./2)*np.linalg.multi_dot([D - func_dist(z,*args), sigma_inv, D - func_dist(z,*args)])+(1./2)*((np.linalg.multi_dot([ones, sigma_inv, D - func_dist(z,*args)]))**2)/(np.linalg.multi_dot([ones, sigma_inv, ones]))
    if type == 'DESI':
        LH = lambda *args: -(1./2)*np.linalg.multi_dot([D - func_dist(z,*args), sigma_inv, D - func_dist(z,*args)])

    return LH

def logL_flatPrior(data, func_dist, infs, sups, sigma=None, type='DESI'):
    '''
    Parameters
    ----------
    data : array
        Data from galaxy survey
        must have the following columns: 'z', 'D_H', 'deltaD_H', 'D_M', 'deltaD_M', 'D_V', 'deltaD_V' if type=='DESI'
        must have the following columns: 'z', 'mu' and the covariance matrix must be given as sigma if type=='SNIa'
    func_dist : function
        Function to calculate the distance,
        must be a lambda function that has as variable the redshift, the paramaeters that we want to estimate;
        the other parameters must be fixed
        (can be any distance function, distance modulus or distance volume);
        must be defines in the notebook
    inf : array
        Lower bounds of the parameters
    sup : array
        Upper bounds of the parameters
    sigma : array
        Covariance matrix
    type : string
        Type of data
        'SNIa' for supernovae
        'DESI' for DESI experiment data
    Returns
    -------
    logL : float
        Log likelihood
    '''

    if type == 'DESI':
        data, sigma = dataDESImanager(data)

    z = data[:,0]
    D = data[:,1]
    n = z.shape[0]

    ones = np.ones(n)
    args = []
    args_sign = signature(func_dist)
    for param in args_sign.parameters.values():
        args.append(str(param))
    args.pop(0)

    print('Arguments of LH: ', args)

    sigma_inv = np.array(np.linalg.inv(sigma))

    flat_prior = lambda *args: 1.0 if ((np.array(args)>infs).all() and (np.array(args)<sups).all()) else 0.0

    if type == 'SNIa':
        LH = lambda *args: (-(1./2)*np.linalg.multi_dot([D - func_dist(z,*args), sigma_inv, D - func_dist(z,*args)])+(1./2)*((np.linalg.multi_dot([ones, sigma_inv, D - func_dist(z,*args)]))**2)/(np.linalg.multi_dot([ones, sigma_inv, ones])))*flat_prior(*args)
    if type == 'DESI':
        LH = lambda *args: (-(1./2)*np.linalg.multi_dot([D - func_dist(z,*args), sigma_inv, D - func_dist(z,*args)]))*flat_prior(*args)

    return LH