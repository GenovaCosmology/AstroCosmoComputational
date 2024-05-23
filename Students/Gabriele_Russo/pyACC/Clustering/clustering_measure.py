import numpy as np
from scipy.spatial import cKDTree
#from tqdm.notebook import tqdm

def get_power_spectrum(delta_x, side, spacing, n_kF=1):
    '''
    Compute the power spectrum from a density field defined on a grid
    '''

    # Get the density field in Fourier space
    delta_k = np.fft.rfftn(delta_x, norm='backward')*spacing**3

    # Compute the power spectrum
    delta_k_sq = np.abs(delta_k)**2

    # Set the Volume
    volume = side**3

    n_cell = int(side//spacing)

    # Set the Fourier Grid
    kx = np.fft.fftfreq(n_cell, d=spacing)*np.pi*2
    ky = np.fft.fftfreq(n_cell, d=spacing)*np.pi*2
    kz = np.fft.rfftfreq(n_cell, d=spacing)*np.pi*2

    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij') # indexing changes the order of the axes of the matrix

    knorm = np.sqrt(KX**2 + KY**2 + KZ**2)



    pks = delta_k_sq.flatten()/volume

    # Binning

    kF = 2*np.pi/side
    kN = 2*np.pi/spacing

    edges = np.arange(kF, kN, n_kF*kF)
    k_bin = (edges[1:] + edges[0:-1])/2

    pk_meas = np.zeros(k_bin.shape)

    for i in range(len(k_bin)):
        k_selection = np.where(knorm.flatten() > edges[i]) & (knorm.flatten() < edges[i+1])[0]
        pk_meas[i] = np.mean(pks[k_selection])

    
    return k_bin, pk_meas

def count_pairs(data_1, r_edges, data_2=None):
    '''
    Count auto/cross pairs between sample 1 and 2
    with a given separation binning

    Caso 1 è coppie AUTO, DD, RR
    mi siedo su una particella e conto quelle attorno a me, ma quando mi muovo conto quelle attorno al nuovo punto
    tra cui quelle che ho già contato
    trucchetto caso auto: conta coppia solo quando indice 1 < indice 2 
    escludiamo anche 1=2 perchè è contare con se stesso (shot noise spazio configurazioni)
    '''

    if data_2 is None: #sto facendo data-data o random-random non data-random
        tree = cKDTree(data_1) #tree è campo contro cui andimao a contare, mentre loop su altro campo
        auto = True
    else:
        tree = cKDTree(data_2)
        auto = False

    pairs = np.zeros(len(r_edges)-1)

    #for i in tqdm(range(len(data_1))):
    for i in range(len(data_1)):

        neighbours_idx = np.array(tree.query_ball_point(data_1[i], r_edges[-1])) # tree ha in sè tutto catalogo,
        #query_ball_point restituisce indici delle particelle vicine a quella che sto considerando 
        # dal centro della particella i esima che stiamo considerando; la funzione rende gli indici dei primi vicini

        if auto:
            cut_neighbours = np.where(neighbours_idx>i)[0]
            neighbours_idx = neighbours_idx[cut_neighbours]

        separations = np.linalg.norm(tree.data[neighbours_idx]-data_1[i], axis= 1)

        pairs += np.histogram(separations, bins=r_edges)[0]



    return pairs