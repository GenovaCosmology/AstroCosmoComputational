import numpy as np


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
