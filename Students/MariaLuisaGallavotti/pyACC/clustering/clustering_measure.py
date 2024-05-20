import numpy as np
from scipy.spatial import cKDTree

def get_power_spectrum(delta_x,side,spacing,n_kF=1):
    '''
    Compute the power spectrum from a desity field on a grid
    '''

    #Grid stuff
    Volume=side**3
    n_cell=int(side//spacing)

    kx=np.fft.fftfreq(n_cell,d=spacing)*2*np.pi
    ky=np.fft.fftfreq(n_cell,d=spacing)*2*np.pi
    kz=np.fft.rfftfreq(n_cell,d=spacing)*2*np.pi

    KX,KY,KZ=np.meshgrid(kx,ky,kz,indexing='ij')
    knorm=np.sqrt(KX**2+KY**2+KZ**2)

    #Get the density field in Fourier space
    delta_k=np.fft.rfftn(delta_x,norm='backward')*spacing**3

    #Compute the power spectrum
    delta_k_sq=np.abs(delta_k)**2
    pks=delta_k_sq.flatten()/Volume

    #Binning
    kF=2*np.pi/side
    kN=2*np.pi/spacing
    edges=np.arange(kF,kN,n_kF*kF)
    k_bin=0.5*(edges[1:]+edges[:-1])

    pk_meas=np.zeros(k_bin.shape)
    for i in range(len(k_bin)):
        k_selection=np.where((knorm.flatten()>edges[i]) & (knorm.flatten()<edges[i+1]))[0]
        pk_meas[i]=np.mean(pks[k_selection])
    
    return k_bin,pk_meas

def count_pairs(data_1,r_edges,data_2=None):
    '''
    Count auto/cross pairs between two datasets, sample 1 and 2, with a given separation binning
    '''
    #data_1 contains all of my particles (it is a catalog)
    #the problem is that the algorithm counts two times the same pair (it is the "shot noise" of configuration space)
    if data_2 is None:
        #tree=field in which I am counting pairs
        tree=cKDTree(data_1)
        auto=True
    else:
        tree=cKDTree(data_2)
        auto=False
    
    pairs=np.zeros(len(r_edges)-1)
    for i in range(len(data_1)):
        neighbours_idx=np.array(tree.query_ball_point(data_1[i],r_edges[-1])) #from the i-th particle, I look for all the neighbors within r_edges[-1]
        if auto:
            cut_neighbours=np.where(neighbours_idx>i)[0]
            neighbours_idx=neighbours_idx[cut_neighbours>i] #I remove the i-th particle
        #now I have the center and all of its neighbors, I want to know the separation
        separations=np.linalg.norm(tree.data[neighbours_idx]-data_1[i],axis=1)
        pairs+=np.histogram(separations,bins=r_edges)[0] #the pair vector is updated with the number of pairs in each bin

    return pairs