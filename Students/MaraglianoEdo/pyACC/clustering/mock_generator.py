import numpy as np

def generate_gaussian_map(pk_func, side, spacing):
    """
    Generates a gaussian map fron an input power spectrum in a given volume

    Parameters: 

    """

    # set the volume
    volume = side**3

    n_cell = int(side//spacing)

    # set the Fourier grid

    kx = np.fft.fftfreq(n_cell, spacing)*np.pi*2
    ky = np.fft.fftfreq(n_cell, spacing)*np.pi*2
    kz = np.fft.rfftfreq(n_cell, spacing)*np.pi*2

    kx = np.fft.fftshift(kx)
    ky = np.fft.fftshift(ky)

    KX, KY, KZ = np.meshgrid(kx,ky,kz, indexing='xy')

    knorm = np.sqrt(KX**2+KY**2+KZ**2)
    pks = pk_func(knorm)

    #generate a random phase
    phase = np.random.uniform(size=pks.shape)*2*np.pi
    #generate a norm
    delta_k_norm = np.random.normal(size=pks.shape)*np.sqrt(pks*volume)

    delta_k_norm[0,0,0]=0

    delta_k = delta_k_norm*(np.cos(phase)+1j*np.sin(phase))

    delta_x = np.fft.irfftn(delta_k, norm='backward')/spacing**3

    return delta_k, delta_x


def generate_log_normal_map(pk_func, side, spacing):
    pass