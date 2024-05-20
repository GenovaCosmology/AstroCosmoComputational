import numpy as np

def generate_gaussian_map(pk_func, side, spacing):
    '''
    Generates a Guaussian map from an input P(k) in a given volume

    Parameters:
    pk_func : function
        The input power spectrum function
    side : int
        The size of the volume
    '''

    # Set the Volume
    volume = side**3

    n_cell = int(side//spacing)

    # Set the Fourier Grid
    kx = np.fft.fftfreq(n_cell, d=spacing)*np.pi*2
    ky = np.fft.fftfreq(n_cell, d=spacing)*np.pi*2
    kz = np.fft.rfftfreq(n_cell, d=spacing)*np.pi*2

    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij') # indexing changes the order of the axes of the matrix

    knorm = np.sqrt(KX**2 + KY**2 + KZ**2)

    # Interpolate the Power Spectrum on the Fourier Grid
    pks = pk_func(knorm)

    # Generate a random phase
    phase = np.random.uniform(size=pks.shape) * np.sqrt(pks * volume)

    delta_k_norm = np.random.normal(size=pks.shape) * np.sqrt(pks * volume)

    delta_k_norm[0,0,0] = 0 # per garantire trasformata sia reale

    delta_k = delta_k_norm * (np.cos(phase) + 1j*np.sin(phase))

    # Compute the field in configuration space

    delta_x = np.fft.irfftn(delta_k, norm='backward')/spacing**3

    return delta_k, delta_x




