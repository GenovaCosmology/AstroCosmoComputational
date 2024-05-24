import numpy as np
from scipy.spatial import cKDTree
from tqdm.notebook import tqdm

def get_power_spectrum(delta_x, side, spacing, n_kF=1):
    """
    Computes the power spectrum of a 3D density field.

    Parameters:
    -----------
    delta_x : ndarray
        A 3D array representing the density contrast field.
    side : float
        The physical size of the simulation box (e.g., in Mpc/h).
    spacing : float
        The spatial resolution of the grid (e.g., in Mpc/h).
    n_kF : int, optional
        The number of fundamental modes per bin for the power spectrum. Default is 1.

    Returns:
    --------
    k_bin : ndarray
        1D array of wavenumber bins' central values.
    pk_meas : ndarray
        1D array of the measured power spectrum values corresponding to `k_bin`.

    Notes:
    ------
    This function assumes periodic boundary conditions and uses the Fast Fourier Transform (FFT) 
    to compute the Fourier transform of the input density field. The power spectrum is calculated 
    by binning the squared magnitudes of the Fourier modes.

    Example:
    --------
    >>> import numpy as np
    >>> delta_x = np.random.randn(64, 64, 64)
    >>> side = 100.0
    >>> spacing = 1.5625
    >>> k_bin, pk_meas = get_power_spectrum(delta_x, side, spacing)
    """
    Volume = side**3
    n_cell = int(side//spacing)

    # set the Fourier grid

    kx = np.fft.fftfreq(n_cell, spacing)*np.pi*2
    ky = np.fft.fftfreq(n_cell, spacing)*np.pi*2
    kz = np.fft.rfftfreq(n_cell, spacing)*np.pi*2

    kx = np.fft.fftshift(kx)
    ky = np.fft.fftshift(ky)

    KX, KY, KZ = np.meshgrid(kx,ky,kz, indexing='xy')

    knorm = np.sqrt(KX**2+KY**2+KZ**2)

    # get the density field in Fourier space
    delta_k = np.fft.rfftn(delta_x, norm='backward')*spacing**3

    # compute the power spectrum
    delta_k_sq = np.abs(delta_k)**2

    pks = delta_k_sq.flatten()/Volume

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
    Count auto-pairs or cross-pairs between two sets of points with a given separation binning.

    Parameters:
    -----------
    data_1 : ndarray
        A 2D array of shape (N1, 3) containing the 3D coordinates of the first set of points.
    r_edges : ndarray
        A 1D array defining the edges of the separation bins.
    data_2 : ndarray, optional
        A 2D array of shape (N2, 3) containing the 3D coordinates of the second set of points.
        If not provided, the function will count auto-pairs within `data_1`.

    Returns:
    --------
    pairs : ndarray
        A 1D array of length `len(r_edges) - 1` containing the counts of pairs for each separation bin.

    Notes:
    ------
    - When computing auto-pairs (i.e., `data_2` is `None`), the function only considers pairs (i, j) with i < j
      to avoid double counting and save computational time.
    - The function uses `cKDTree` from `scipy.spatial` to efficiently find neighboring points within the specified
      separation bins.

    Example:
    --------
    >>> import numpy as np
    >>> data_1 = np.random.rand(100, 3)
    >>> r_edges = np.linspace(0, 1, 11)
    >>> pairs = count_pairs(data_1, r_edges)
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


def get_xi_LS(data_1, r_edges, data_2):
    """
    Compute the Landy-Szalay estimator for the two-point correlation function.

    Parameters:
    -----------
    data_1 : ndarray
        A 2D array of shape (N1, 3) containing the 3D coordinates of the first set of points.
    r_edges : ndarray
        A 1D array defining the edges of the separation bins.
    data_2 : ndarray
        A 2D array of shape (N2, 3) containing the 3D coordinates of the second set of points.
    
    Returns:
    --------
    r : ndarray
        1D array of the central values of the separation bins.
    xi_ls : ndarray
        1D array of the Landy-Szalay estimator values corresponding to `r`.
    
    Notes:
    ------
    The Landy-Szalay estimator is defined as:

    .. math::
        \\xi_{LS}(r) = \\frac{DD(r) - 2DR(r) + RR(r)}{RR(r)}

    where DD(r), DR(r), and RR(r) are the counts of data-data, data-random, and random-random pairs
    within a given separation bin, respectively.

    Example:
    --------
    >>> import numpy as np
    >>> data_1 = np.random.rand(100, 3)
    >>> data_2 = np.random.rand(100, 3)
    >>> r_edges = np.linspace(0, 1, 11)
    >>> r, xi_ls = get_xi_LS(data_1, r_edges, data_2)
    """

    # count pairs for data_1
    DD = count_pairs(data_1, r_edges)/(len(data_1)*(len(data_1)-1)/2)

    # count pairs for data_2
    RR = count_pairs(data_2, r_edges)/(len(data_2)*(len(data_2)-1)/2)

    # count cross pairs
    DR = count_pairs(data_1, r_edges, data_2)/(len(data_1)*len(data_2))

    r = (r_edges[1:] + r_edges[:-1]) / 2
    xi_ls = (DD - 2 * DR + RR) / RR

    return r, xi_ls