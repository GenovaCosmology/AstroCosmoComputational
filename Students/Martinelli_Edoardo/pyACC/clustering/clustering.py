import numpy as np
import scipy as sp
from tqdm.notebook import tqdm

def PowerSpectrum(dist,L,Nkbin=100):
    '''
    This function calculates the power spectrum of a distribution of points in a homogeneous and isotropic space.
    
    Parameters:
    - dist: distribution of points on uniform N-dimensional grid (numpy array)
    - L: size of the box (float)
    - Nkbin: number of bins in k-space (int)
    
    Returns:
    - k: wavenumbers (numpy array)
    - Pk: power spectrum (numpy array)
    '''

    dim=len(list(dist.shape))
    V=L**dim
    dist=np.array(dist)
    N=dist.shape[0]

    # Calculate the Fourier transform of the distribution
    ft=np.fft.fftn(dist)
    # Calculate the power spectrum
    ps=np.real(ft*np.conj(ft)*V)
    PS=ps.flatten()

    # Calculate the wavenumbers
    arr=np.linspace(0,L,N)
    ki=2*np.pi*(1/L)*arr
    ks=[ki]*dim
    K=np.meshgrid(*ks)
    Kmod2=0
    for i in range(dim):
        Kmod2+=K[i]**2
    Kmod=np.sqrt(Kmod2)
    kmods=Kmod.flatten()

    kbin=np.linspace(np.min(kmods),np.max(kmods),Nkbin)
    Pk=[]
    k=[]
    for i,kis in enumerate(kbin):
        pks=[]
        if i==kbin.size-1:
            break
        for j,kj in enumerate(kmods):
            if kj>=kis and kj<kbin[i+1]:
                pks.append(PS[j])
        if len(pks)!=0:
            Pk.append(np.mean(pks))
            k.append(kis)
    return np.array(k),np.array(Pk)


def count_pairs(D,R,rmax,rmin=0,Nbins=100,flag='all'):
    '''
    This function calculates the pair counts between two distributions of points.

    Parameters:
    - D: distribution of points (numpy array) (size N x dim)
    - R: distribution of points (numpy array) (size M x dim)
    - rmax: maximum separation (float)
    - Nbins: number of bins in separation (int)
    - rmin: minimum separation (float)
    - flag: flag to calculate the pair counts between D-D, R-R, D-R or all (string)

    Returns: (for all returns, flag='all')
    - r: separation (numpy array)
    - DD: pair counts between D-D (numpy array); flag='DD'
    - RR: pair counts between R-R (numpy array); flag='RR'
    - DR: pair counts between D-R (numpy array); flag='DR'
    '''

    r=np.linspace(rmin,rmax,Nbins)
    delR=np.abs(rmax-rmin)/Nbins

    treeD=sp.spatial.cKDTree(D)
    treeR=sp.spatial.cKDTree(R)

    if flag=='DD':
        # Calculate the pair counts between D-D
        DD=np.zeros(Nbins)
        indListDD=list(treeD.query_ball_point(D,rmax))
        for i in tqdm(range(len(indListDD)),desc=' Calculating DD... '):
            for j in range (len(indListDD[i])):
                if indListDD[i][j]>i:#if (D[i]!=D[indListDD[i][j]]).any():
                    DD[int(np.linalg.norm(D[i]-D[indListDD[i][j]])/delR)]+=1
        return r,DD/(len(D)*(len(D)-1)/2)
         
    if flag=='RR':
        # Calculate the pair counts between R-R
        RR=np.zeros(Nbins)
        indListRR=list(treeR.query_ball_point(R,rmax))
        for i in tqdm(range(len(indListRR)),desc=' Calculating RR... '):
            for j in range (len(indListRR[i])):
                if indListRR[i][j]>i:#if (R[i]!=R[indListRR[i][j]]).any():
                    RR[int(np.linalg.norm(R[i]-R[indListRR[i][j]])/delR)]+=1
        return r,RR/(len(R)*(len(R)-1)/2)

    if flag=='DR':
        # Calculate the pair counts between D-R
        DR=np.zeros(Nbins)
        indListDR=list(treeR.query_ball_point(D,rmax))
        for i in tqdm(range(len(indListDR)),desc=' Calculating DR... '):
            for j in range (len(indListDR[i])):
                DR[int(np.linalg.norm(D[i]-R[indListDR[i][j]])/delR)]+=1
        return r,DR/(len(D)*len(R))

    if flag=='all':
        # Calculate the pair counts between D-D
        DD=np.zeros(Nbins)
        indListDD=list(treeD.query_ball_point(D,rmax))
        for i in tqdm(range(len(indListDD)),desc=' Calculating DD... '):
            for j in range (len(indListDD[i])):
                if indListDD[i][j]>i:#if (D[i]!=D[indListDD[i][j]]).any():
                    DD[int(np.linalg.norm(D[i]-D[indListDD[i][j]])/delR)]+=1

        # Calculate the pair counts between R-R
        RR=np.zeros(Nbins)
        indListRR=list(treeR.query_ball_point(R,rmax))
        for i in tqdm(range(len(indListRR)),desc=' Calculating RR... '):
            for j in range (len(indListRR[i])):
                if indListRR[i][j]>i:#if (R[i]!=R[indListRR[i][j]]).any():
                    RR[int(np.linalg.norm(R[i]-R[indListRR[i][j]])/delR)]+=1

        # Calculate the pair counts between D-R
        DR=np.zeros(Nbins)
        indListDR=list(treeR.query_ball_point(D,rmax))
        for i in tqdm(range(len(indListDR)),desc=' Calculating DR... '):
            for j in range (len(indListDR[i])):
                DR[int(np.linalg.norm(D[i]-R[indListDR[i][j]])/delR)]+=1

        return r,DD/(len(D)*(len(D)-1)/2),RR/(len(R)*(len(R)-1)/2),DR/(len(D)*len(R))
    

def estimated_correlation_function(D,R,rmax,rmin=0,Nbins=100,type='LS'):

    '''
    This function calculates the correlation function between two distributions of points.

    Parameters:
    - D: distribution of points (numpy array) (size N x dim)
    - R: distribution of points (numpy array) (size M x dim)
    - rmax: maximum separation (float)
    - Nbins: number of bins in separation (int)
    - rmin: minimum separation (float)
    - type: type of correlation function (string)
            - 'LS' : Landy-Szalay estimator
            - 'Ham': Hamilton estimator
            - 'PH' : Peebles & Hauser estimator
            - 'DP' : Davis & Peebles estimator
            - 'H'  : Hewett estimator
    Returns:
    - r: separation (numpy array)
    - xi: correlation function (numpy array)
    '''
    
    if type=='LS':
        r,DD,RR,DR=count_pairs(D,R,rmax=rmax,rmin=rmin,Nbins=Nbins,flag='all')
        xi=(DD-2*DR+RR)/RR
    if type=='Ham':
        r,DD,RR,DR=count_pairs(D,R,rmax=rmax,rmin=rmin,Nbins=Nbins,flag='all')
        xi=(DD*RR)/(DR**2)-1
    if type=='PH':
        r,DD=count_pairs(D,R,rmax=rmax,rmin=rmin,Nbins=Nbins,flag='DD')
        RR=count_pairs(R,R,rmax=rmax,rmin=rmin,Nbins=Nbins,flag='RR')[1]
        xi=(DD/RR)-1
    if type=='DP':
        r,DD=count_pairs(D,R,rmax=rmax,rmin=rmin,Nbins=Nbins,flag='DD')
        DR=count_pairs(R,R,rmax=rmax,rmin=rmin,Nbins=Nbins,flag='DR')[1]
        xi=(DD/DR)-1
    if type=='H':
        r,DD=count_pairs(D,R,rmax=rmax,rmin=rmin,Nbins=Nbins,flag='DD')
        DR=count_pairs(R,R,rmax=rmax,rmin=rmin,Nbins=Nbins,flag='DR')[1]
        xi=(DD-DR)/DR
    return r,xi


