import numpy as np
import scipy as sp
from tqdm.notebook import tqdm

def PowerSpectrum(f, L, k_n):
    '''
    Function to calculate the power spectrum

    Parameters:
    f: array
        function defined on a NxNxN grid of volume LxLxL
    N: float
        number of points in each direction of the grid
    L: float
        lenght in each dimension of the volume considered
    k_n: int
        number of bins in wich the power spectrum will be calculated

    Return:
    k_array: array
        array of k values
    P_k: array
        binned power spectrum
    '''
    N = f.shape[0]
    dim = len(list(f.shape))
    V = L**dim

    FT_grid = np.fft.fftn(f)
    P = np.real(FT_grid*np.conjugate(FT_grid))*V

    n_array = np.linspace(0,L,N)

    k = (2*np.pi/L)*n_array

    k_list = [k]*dim
    
    K_mesh = np.meshgrid(*k_list)

    grid_module2 = 0.0

    for i in range(dim):
        grid_module2 += K_mesh[i]**2

    grid_module = np.sqrt(grid_module2)

    k_min = np.min(grid_module)
    k_max = np.max(grid_module) + 1
    k_array = np.linspace(k_min, k_max, k_n)

    k_flatten = grid_module.flatten()
    P_flatten = P.flatten()

    P_binned = [[] for i in range(k_n)]

    for i, kk in enumerate(k_flatten):
        for j in range(k_n-1):
            
            if k_array[j]<=kk<k_array[j+1]:
                P_binned[j].append(P_flatten[i])

    P_k = []
    for i in range(k_n):
        P_k.append(np.mean(P_binned[i]))

    return k_array, P_k


def counting_pairs(D, R, r_max, r_min=0, n_bin=100, output = 'all'):
    '''
    Parameters
    data: array
        array of points in a 3D space nxm (n number of point and m space dimension)
    random: array
        array of random points in a 3D space
    r_max: float
        maximum distance to be considered
    r_min: float
        minimum distance to be considered
    n_bin: int
        number of bins to be used
    
    Return
    r: array
        array of distances
    DD: array
        number of pairs in each bin for data-data pairs
    DR: array
        number of pairs in each bin for data-random pairs
    RR: array
        number of pairs in each bin for random-random pairs
    '''

    r = np.linspace(r_min, r_max, n_bin)
    delta_r = (r_max - r_min)/n_bin
    tree_D = sp.spatial.cKDTree(D)
    tree_R = sp.spatial.cKDTree(R)

    if output == 'DD':
        DD = np.zeros(n_bin)
        indDD = list(tree_D.query_ball_point(D, r_max))
        for i in tqdm(range(len(indDD)), desc='DD: '):
            for j in range(len(indDD[i])):
                if i<indDD[i][j]:
                    DD[int(np.linalg.norm(D[i]-D[indDD[i][j]])/delta_r)] += 1
                           
        return r, DD/(len(D)*(len(D)-1)/2) 

    if output == 'RR':
        RR = np.zeros(n_bin)
        indRR = list(tree_R.query_ball_point(R, r_max))
        for i in tqdm(range(len(indRR)), desc='RR: '):
            for j in range(len(indRR[i])):
                if i<indRR[i][j]:
                    RR[int(np.linalg.norm(R[i]-R[indRR[i][j]])/delta_r)] += 1
                          
        return r, RR/(len(R)*(len(R)-1)/2) 
        

    if output == 'DR':
        DR = np.zeros(n_bin)
        indDR = list(tree_R.query_ball_point(D, r_max))
        for i in tqdm(range(len(indDR)), desc='DR: '):
            for j in range(len(indDR[i])):
                DR[int(np.linalg.norm(D[i]-R[indDR[i][j]])/delta_r)] += 1                
        
        return r, DR/(len(D)*len(R))


    if output == 'all':
        DD = np.zeros(n_bin)
        indDD = list(tree_D.query_ball_point(D, r_max))
        for i in tqdm(range(len(indDD)), desc='DD: '):
            for j in range(len(indDD[i])):
                if i<indDD[i][j]:
                    DD[int(np.linalg.norm(D[i]-D[indDD[i][j]])/delta_r)] += 1
                    
        RR = np.zeros(n_bin)
        indRR = list(tree_R.query_ball_point(R, r_max))
        for i in tqdm(range(len(indRR)), desc='RR: '):
            for j in range(len(indRR[i])):
                if i<indRR[i][j]:
                    RR[int(np.linalg.norm(R[i]-R[indRR[i][j]])/delta_r)] += 1
                                       
        DR = np.zeros(n_bin)
        indDR = list(tree_R.query_ball_point(D, r_max))
        for i in tqdm(range(len(indDR)), desc='DR: '):
            for j in range(len(indDR[i])):
                DR[int(np.linalg.norm(D[i]-R[indDR[i][j]])/delta_r)] += 1
                
        return r, DD/(len(D)*(len(D)-1)/2), RR/(len(R)*(len(R)-1)/2), DR/(len(D)*len(R))
    

def correlation_function(D, R, r_max, r_min=0, n_bin=100, type='LS'):
    '''
    Parameters
    data: array
        array of points in a 3D space nxm (n number of point and m space dimension)
    random: array
        array of random points in a 3D space
    r_max: float
        maximum distance to be considered
    r_min: float
        minimum distance to be considered
    n_bin: int
        number of bins to be used
    
    Return
    r: array
        array of distances
    xi: array
        correlation function
    '''
    r, DD, RR, DR = counting_pairs(D, R, r_max=r_max, r_min=r_min, n_bin=n_bin, output='all')    

    if type == 'PH':
        xi = (DD/RR)-1
    
    if type == 'Hw':
        xi = (DD-DR)/RR
    
    if type == 'DP':
        xi = (DD/DR)-1
    
    if type == 'Hm':
        xi = (DD*RR)/(DR**2)-1

    if type == 'LS':
        xi = (DD-2*DR+RR)/RR

    return r, xi