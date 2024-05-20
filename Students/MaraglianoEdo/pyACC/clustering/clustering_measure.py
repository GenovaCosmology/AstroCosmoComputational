import numpy as np
from scipy.spatial import cKDTree
from tqdm.notebook import tqdm

def get_power_spectrum(delta_x, side, spacing, n_kF=1):

    Volume = side**3
    n_cell = int(side//spacing)

    # set the Fourier grid

    kx = np.fft.fftfreq(n_cell, spacing)*np.pi*2
    ky = np.fft.fftfreq(n_cell, spacing)*np.pi*2
    kz = np.fft.rfftfreq(n_cell, spacing)*np.pi*2

    kx = np.fft.fftshift(kx)
    ky = np.fft.fftshift(ky)

    KX, KY, KZ = np.meshgrid(kx,ky,kz, indexing='ij')

    knorm = np.sqrt(KX**2+KY**2+KZ**2)

    # get the density field in Fourier space
    delta_k = np.fft.rfftn(delta_x)/Volume

    # compute the power spectrum
    delta_k_sq = Volume*np.abs(delta_k)**2

    pks = delta_k_sq.flatten()

    # binning

    kF = 2*np.pi/side
    kN = 2*np.pi/spacing
    edges = np.arange(kF,kN, n_kF*kF)
    k_bin = (edges[1:]+edges[0:-1])/2

    pk_meas = np.zeros(k_bin.shape)

    # Loop through each bin edge
    for i in range(len(k_bin)-1):
        # Find indices where the values of k_norm fall within the range defined by the bin edges
        k_selection = np.where((knorm.flatten() >= edges[i]) & (knorm.flatten() < edges[i + 1]))[0]
        
        # Calculate the mean value of Pk within the bin
        pk_meas[i] = np.mean(pks.flatten()[k_selection])

    return k_bin, pk_meas



def count_pairs(data_1, r_edges, data_2 = None ):
    """
    Count auto/cross pairs between saple 1 and 2
    with a given separation binning
    """

    # when I am computing auto pairs, I only cycle on i1 < i2
    # otherwise I am countig each pair twice, wasting computational time
    # tree = second D in DD, second R in RR ecc
    # loop on the first letter
    if data_2 is None:
        tree = cKDTree(data_1)
        auto = True
    else:
        tree = cKDTree(data_2)
        auto = False

    pairs = np.zeros(len(r_edges)-1)

    for i in tqdm(range(len(data_1))):
        
        # get index of the neighbours particles of data_1[i] with radius the last r_edges
        neighbours_idx = np.array(tree.query_ball_point(data_1[i], r_edges[-1]))
        cut_neighbours = np.where(neighbours_idx>i)[0]

        if(auto):
            neighbours_idx = neighbours_idx[cut_neighbours]
            separations = np.linalg.norm(tree.data[neighbours_idx]-data_1[i], axis=1)
            pairs += np.histogram(separations, bins=r_edges)[0]

    return pairs