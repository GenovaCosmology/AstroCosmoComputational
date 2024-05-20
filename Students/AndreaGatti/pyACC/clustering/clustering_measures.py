import numpy as np
from scipy.spatial import cKDTree

def get_power_spectrum(delta_x, side, spacing, n_kF=1):
    """
    Compute the power spectrum froma  density field defiened on a grid.

    """

    #Get the density field in Fourier space
    delta_k = np.fft.rfftn(delta_x, norm="backward")*spacing**3

    #Compute the power spectrum
    delta_k_sq = np.abs(delta_k)**2

    #Set the volume
    Volume = side**3

    n_cell = int(side//spacing)
    #Set the Fourier grid
    kx = np.fft.fftfreq(n_cell, d=spacing)*np.pi*2
    ky = np.fft.fftfreq(n_cell, d=spacing)*np.pi*2
    kz = np.fft.rfftfreq(n_cell, d=spacing)*np.pi*2

    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='xy')

    knorm = np.sqrt(KX**2 + KY**2 + KZ**2)

    #Interpolate the power spectrum on the grid
    pks = delta_k_sq.flatten()/Volume 


    #Binning
    kF = 2*np.pi/side
    kN = 2*np.pi/spacing
    edges = np.arange(kF, kN, kF*n_kF)

    k_bin = (edges[1:] + edges[:-1])/2
    pk_meas = np.zeros(k_bin.shape)
    for i in range(len(k_bin)):
        k_selection = np.where((knorm.flatten() > edges[i]) & (knorm.flatten() < edges[i+1]))
        pk_meas[i] = np.mean(pks[k_selection])

    return k_bin, pk_meas


def count_pairs(data_1, r_edges, data_2=None):
    """
    Count auto/croos pairs between samples in data_1 and data_2 with a given separation binning.

    """
    #scritto alla lavagna: xi(r)=(DD(r)-2DR(r)+RR(r))/RR(r)

    if data_2 is None:
        tree = cKDTree(data_1)
        auto = True
    else:
        tree = cKDTree(data_2)
        auto = False

    pairs = np.zeros(len(r_edges)-1)


    for i in tqdm(range(len(data_1))):
        neighbours_idx = tree.query_ball_point(data_1[i], r_edges[-1]) #r_edges[-1] is the maximum separation, and la i-esima particella è la centro

        if auto:
            cut_neighbours = np.where(neighbours_idx > i)
            neighbours_idx = neighbours_idx[neighbours_idx > i]

        separations  = np.linalg.norm(tree.data_1[neighbours_idx]-data_1[i], axis=1)

        pairs += np.histogram(separations, bins=r_edges)[0]
    return pairs