import numpy as np

def generate_gaussian_map(pK_func, side, spacing):
    """
    Generate a mock map with a gaussian distribution of points.

    """

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
    pks = pK_func(knorm)

    #Generate a random field
    phase = np.random.uniform(size=pks.shape)*2*np.pi

    delta_k_norm= np.random.normal(size=pks.shape) * np.sqrt(pks*Volume)

    delta_k_norm[0,0,0] = 0

    delta_k = delta_k_norm * (np.cos(phase) + 1j*np.sin(phase))

    #Compute the field in real space
    delta_x = np.fft.irfftn(delta_k, norm="backward")/spacing**3

    return delta_k, delta_x