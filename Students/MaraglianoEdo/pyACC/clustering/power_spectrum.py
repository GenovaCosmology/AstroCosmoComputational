import numpy as np


def compute_power_spectrum(real_coord, delta):
    # Compute grid spacing for each dimension
    spacings = [coord[1] - coord[0] for coord in real_coord]

    # Check that real_coord is a cubic box (tbd more general case)
    lengths = np.array([len(coord) for coord in real_coord])
    if not np.all(lengths == lengths[0]):
        raise ValueError('the grid must be a box: x,y,z should have the same dimension')

    # Compute volume
    vol = np.prod([len(axis) for axis in real_coord])

    # Precompute FFT frequencies for each dimension
    freqs = [np.fft.fftfreq(len(coord), spacing) for coord, spacing in zip(real_coord, spacings)]

    # Compute the 3D Fourier frequencies mesh
    kxs, kys, kzs = np.meshgrid(*freqs, indexing='ij')

    # Compute the 3D Fourier transform of the density field
    delta_transform = np.fft.fftn(delta)

    # Compute the normalized power spectrum
    Pk = np.abs(delta_transform)**2 / vol  

    # Compute the norm of wavevectors
    k_norm = np.sqrt(kxs**2 + kys**2 + kzs**2)

    # Define the edges for the wavevector bins
    kN = 2 * np.pi / spacings[0]              # max frequency based on spacing
    kF = 2 * np.pi / len(real_coord[0])       # min frequency based on side length

    # Define the edges for the wavevector bins
    k_bin_edges = np.arange(0, k_norm.max() + kF, kF)    # Edges of the bins in abs(k)

    # Initialize arrays to store the bin values and the mean values of k_norm and Pk within each bin
    k_bin = np.zeros(len(k_bin_edges) - 1)
    pk_meas = np.zeros(len(k_bin_edges) - 1)

    # Loop through each bin edge
    for i in range(len(k_bin_edges)-1):
        # Find indices where the values of k_norm fall within the range defined by the bin edges
        indices_in_bin = np.where((k_norm.flatten() >= k_bin_edges[i]) & (k_norm.flatten() < k_bin_edges[i + 1]))[0]
        
        # Calculate the mean value of k_norm within the bin
        k_bin[i] = np.mean(k_norm.flatten()[indices_in_bin])
        
        # Calculate the mean value of Pk within the bin
        pk_meas[i] = np.mean(Pk.flatten()[indices_in_bin])

    return pk_meas,  k_bin